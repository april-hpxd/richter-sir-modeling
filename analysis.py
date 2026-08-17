"""Decision-support analysis for epidemic intervention planning.

This module turns the simulator from a descriptive animation tool into a
research-oriented decision-support system. It evaluates intervention
strategies across repeated seeds, ranks the interventions by expected benefit,
and quantifies the network role of each city using graph-theoretic metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from config import Config
from regional_simulation import RegionalSimulation


@dataclass
class InterventionSpec:
    """A single intervention scenario to evaluate."""

    name: str
    kind: str
    city_ids: Optional[Sequence[int]] = None
    factor: float = 1.0
    connection: Optional[Tuple[int, int]] = None
    threshold_day: int = 0
    threshold_prevalence: float = 0.1


def summarize_simulation(sim: RegionalSimulation) -> Dict[str, float]:
    """Extract a compact set of outcome metrics from one completed run."""
    summary = sim.regional_summary()
    peak_day = max(sim.history, key=lambda h: h["total_infectious"])
    active_days = [h["day"] for h in sim.history if h["total_exposed"] + h["total_infectious"] > 0]
    return {
        "average_arrival_delay": float(summary["average_arrival_delay"]),
        "cities_reached": float(summary["cities_reached"]),
        "peak_regional_infectious": float(peak_day["total_infectious"]),
        "peak_regional_infectious_day": float(peak_day["day"]),
        "epidemic_duration": float(max(active_days) if active_days else 0),
        "attack_rate": float(summary["regional_attack_rate"]),
        "imported_infections": float(summary["imported_infections"]),
        "total_infections": float(summary["total_regional_infections"]),
        "first_infection_days": [float(v) for v in summary["city_first_infection_days"]],
    }


def _apply_intervention(sim: RegionalSimulation, intervention: InterventionSpec, day: int) -> None:
    """Modify the travel layer in place to represent the chosen intervention."""
    matrix = sim.travel.matrix
    if intervention.kind in {"isolate_city", "isolate_cities"}:
        for city_id in intervention.city_ids or []:
            matrix[city_id, :] = 0.0
            matrix[:, city_id] = 0.0
            matrix[city_id, city_id] = 0.0
        return

    if intervention.kind == "reduce_travel_between":
        if intervention.connection is not None:
            src, dst = intervention.connection
            matrix[src, dst] *= intervention.factor
            matrix[dst, src] *= intervention.factor
        return

    if intervention.kind == "reduce_travel_globally":
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i != j:
                    matrix[i, j] *= intervention.factor
        return

    if intervention.kind == "remove_connection":
        if intervention.connection is not None:
            src, dst = intervention.connection
            matrix[src, dst] = 0.0
            matrix[dst, src] = 0.0
        return

    if intervention.kind == "quarantine_after_threshold":
        if day < intervention.threshold_day:
            return
        for city_id in intervention.city_ids or []:
            city = sim.cities[city_id]
            if not city.history:
                continue
            latest = city.history[-1]
            prevalence = (latest.exposed + latest.infectious) / max(1, city.config.population_size)
            if prevalence >= intervention.threshold_prevalence:
                matrix[city_id, :] = 0.0
                matrix[:, city_id] = 0.0
                matrix[city_id, city_id] = 0.0
        return


def _run_intervention_simulation(base_config: Config, intervention: InterventionSpec, seed: int) -> RegionalSimulation:
    """Run one simulation with the intervention applied during the run."""
    sim = RegionalSimulation(base_config.with_overrides(random_seed=seed))
    _apply_intervention(sim, intervention, day=0)

    for day in range(1, base_config.simulation_days + 1):
        if intervention.kind == "quarantine_after_threshold":
            _apply_intervention(sim, intervention, day=day)
        sim.step()
        if not sim.has_active_disease_or_travel():
            break
    return sim


def mean_and_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    """Return mean, standard deviation, and 95% CI half-width."""
    values_array = np.array(values, dtype=float)
    if len(values_array) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values_array.mean())
    if len(values_array) == 1:
        return mean, 0.0, 0.0
    std = float(values_array.std(ddof=1))
    stderr = std / np.sqrt(len(values_array))
    half_width = 1.96 * stderr
    return mean, std, half_width


def evaluate_interventions(
    base_config: Config,
    interventions: Sequence[InterventionSpec],
    num_runs: int = 3,
    base_seed: int = 0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Evaluate a set of interventions across repeated seeds and rank them."""
    baseline_metrics: List[Dict[str, float]] = []
    for seed_offset in range(num_runs):
        seed = base_seed + seed_offset
        sim = RegionalSimulation(base_config.with_overrides(random_seed=seed))
        sim.run(verbose=False)
        baseline_metrics.append(summarize_simulation(sim))

    ranked_interventions: List[Dict[str, Any]] = []
    for intervention in interventions:
        intervention_metrics: List[Dict[str, float]] = []
        for seed_offset in range(num_runs):
            seed = base_seed + seed_offset
            sim = _run_intervention_simulation(base_config, intervention, seed=seed)
            intervention_metrics.append(summarize_simulation(sim))
            if verbose:
                print(f"[{intervention.name}] seed {seed}: total infections={intervention_metrics[-1]['total_infections']:.0f}")

        mean_baseline = float(np.mean([m["total_infections"] for m in baseline_metrics]))
        intervention_totals = [m["total_infections"] for m in intervention_metrics]
        reductions = [mean_baseline - total for total in intervention_totals]
        reduction_pct = [((mean_baseline - total) / mean_baseline * 100.0) if mean_baseline > 0 else 0.0 for total in intervention_totals]
        mean_reduction, std_reduction, ci95 = mean_and_ci(reduction_pct)
        mean_delay_gain = float(np.mean([baseline_metrics[i]["average_arrival_delay"] - intervention_metrics[i]["average_arrival_delay"] for i in range(len(baseline_metrics))]))

        ranked_interventions.append(
            {
                "name": intervention.name,
                "kind": intervention.kind,
                "reduction_infections": max(0.0, float(mean_baseline - float(np.mean(intervention_totals)))),
                "reduction_percent": float(mean_reduction),
                "reduction_std": float(std_reduction),
                "reduction_ci95": float(ci95),
                "delay_gain_days": float(mean_delay_gain),
                "mean_total_infections": float(np.mean(intervention_totals)),
                "mean_attack_rate": float(np.mean([m["attack_rate"] for m in intervention_metrics])),
                "mean_peak_infectious": float(np.mean([m["peak_regional_infectious"] for m in intervention_metrics])),
                "raw_results": intervention_metrics,
            }
        )

    ranked_interventions.sort(key=lambda row: row["reduction_percent"], reverse=True)
    return {
        "baseline_metrics": baseline_metrics,
        "interventions": interventions,
        "ranked_interventions": ranked_interventions,
        "summary": {
            "baseline_mean_total_infections": float(np.mean([m["total_infections"] for m in baseline_metrics])),
            "baseline_mean_attack_rate": float(np.mean([m["attack_rate"] for m in baseline_metrics])),
        },
    }


def analyze_network_importance(sim: RegionalSimulation) -> Dict[str, Any]:
    """Compare graph-theoretic centrality with epidemic importance."""
    city_metrics: List[Dict[str, Any]] = []
    for city in sim.cities:
        graph = city.network
        if graph is None:
            graph = nx.Graph()
            graph.add_nodes_from(range(city.config.population_size))

        degree = nx.degree_centrality(graph)
        betweenness = nx.betweenness_centrality(graph, normalized=True)
        closeness = nx.closeness_centrality(graph)
        communities = list(nx.algorithms.community.greedy_modularity_communities(graph)) if graph.number_of_nodes() > 1 else [set(range(graph.number_of_nodes()))]

        summary = sim.city_summary(city.id)
        metrics = {
            "city_id": city.id,
            "population": int(summary["population"]),
            "degree_centrality": float(np.mean(list(degree.values()))),
            "betweenness_centrality": float(np.mean(list(betweenness.values()))),
            "closeness_centrality": float(np.mean(list(closeness.values()))),
            "community_count": len(communities),
            "clustering_coefficient": float(nx.average_clustering(graph)),
            "attack_rate": float(summary["attack_rate"]),
            "imported_infections": float(summary["imported_infections"]),
            "peak_infectious": float(summary["peak_infectious"]),
            "first_infection_day": float(summary["first_infection_day"]),
            "epidemic_duration_days": float(summary["epidemic_duration_days"]),
        }

        # Clustered-contact-model cities additionally report their explicit
        # cluster structure (distinct from the generic community detection
        # above, which infers communities from any graph).
        cluster_of = getattr(city.engine.contact_model, "cluster_of", None)
        if cluster_of is not None:
            num_clusters = int(cluster_of.max()) + 1
            metrics["num_clusters"] = num_clusters
            metrics["avg_cluster_size"] = city.config.population_size / num_clusters
            metrics["cross_cluster_edge_fraction"] = (
                city.engine.contact_model.cross_cluster_edge_fraction())

        city_metrics.append(metrics)

    attack_rates = np.array([m["attack_rate"] for m in city_metrics], dtype=float)
    degree_scores = np.array([m["degree_centrality"] for m in city_metrics], dtype=float)
    betweenness_scores = np.array([m["betweenness_centrality"] for m in city_metrics], dtype=float)
    imported_scores = np.array([m["imported_infections"] for m in city_metrics], dtype=float)

    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    city_rankings = sorted(
        city_metrics,
        key=lambda row: (
            row["imported_infections"],
            row["attack_rate"],
            row["peak_infectious"],
            -row["first_infection_day"],
        ),
        reverse=True,
    )

    return {
        "city_metrics": city_metrics,
        "city_rankings": city_rankings,
        "correlations": {
            "degree_vs_attack_rate": _corr(degree_scores, attack_rates),
            "betweenness_vs_imported": _corr(betweenness_scores, imported_scores),
            "degree_vs_imported": _corr(degree_scores, imported_scores),
        },
        "research_summary": {
            "largest_outbreak_source": city_rankings[0]["city_id"] if city_rankings else None,
            "most_influential_city": city_rankings[0]["city_id"] if city_rankings else None,
            "largest_transmission_hub": max(city_metrics, key=lambda row: row["betweenness_centrality"])["city_id"] if city_metrics else None,
        },
    }


def print_decision_support_report(report: Dict[str, Any], config: Optional[Config] = None) -> None:
    """Print a concise research-oriented decision-support summary."""
    print("\n" + "=" * 72)
    print("  DECISION SUPPORT SUMMARY")
    print("=" * 72)
    if config is not None:
        print(f"  Cities: {config.num_cities()}  Populations: {config.city_sizes()}")

    ranked = report.get("ranked_interventions", [])
    if ranked:
        print("\n  Intervention ranking (expected change in regional infections):")
        for idx, row in enumerate(ranked, start=1):
            if row["reduction_percent"] >= 0:
                change_text = f"{row['reduction_percent']:.1f}% reduction"
            else:
                change_text = f"{-row['reduction_percent']:.1f}% increase"
            print(f"    {idx}. {row['name']} -> {change_text} "
                  f"(±{row['reduction_ci95']:.1f}%, {row['delay_gain_days']:.1f} day delay gain)")

    network_report = report.get("network_analysis", {})
    if network_report:
        city_rankings = network_report.get("city_rankings", [])
        if city_rankings:
            top_city = city_rankings[0]
            print(f"\n  Best city to isolate: City {top_city['city_id']}")
            print(f"  Largest transmission hub: City {network_report['research_summary']['largest_transmission_hub']}")
            print(f"  Correlation (degree vs attack rate): {network_report['correlations']['degree_vs_attack_rate']:.2f}")

    print("=" * 72 + "\n")
