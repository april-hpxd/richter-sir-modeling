"""Repeated-simulation experiment framework for research figures.

Runs the same regional configuration many times under different random seeds
and aggregates the outcomes (mean and standard deviation). 

The framework only *reads* completed simulations -- it adds no disease, travel,
or visualization behaviour of its own.
"""

from __future__ import annotations

import csv
import dataclasses
import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from analysis import mean_and_ci
from config import Config
from regional_simulation import RegionalSimulation


@dataclass
class RunResult:
    """Metrics extracted from a single completed regional run.

    Fields are grouped to match the "identification / model / network /
    outcomes" structure used throughout the project's research documentation
    (see README.md and CHANGELOG_RESEARCH.md), so that one CSV row is enough
    to reproduce or interpret that single run without cross-referencing code.
    """

    # -- Identification --
    seed: int
    scenario: str
    experiment_name: str
    replicate: int

    # -- Model / disease / mobility parameters --
    population: int
    initial_infected: int
    contact_model: str
    infection_probability: float
    incubation_days: int
    infectious_days: int
    simulation_days: int
    daily_travel_rate: float
    travel_fraction: float
    cluster_count: int
    within_cluster_probability: float
    average_contacts: float
    std_contacts: float

    # -- Network structure (city 0 -- see network_report() caveat for
    #    daily-resampled models: a representative single-day snapshot, not
    #    an average over the run) --
    network_node_count: float
    network_edge_count: float
    network_mean_degree: float
    network_std_degree: float
    network_clustering_coefficient: float
    network_num_connected_components: float
    network_largest_component_size: float
    network_average_shortest_path_length: float
    network_path_length_computed: bool

    # -- Outcomes --
    average_arrival_delay: float
    cities_reached: int
    peak_regional_infectious: int
    peak_regional_infectious_day: int
    epidemic_duration: int
    duration_censored: bool
    total_infected: int
    attack_rate: float
    imported_infections: int
    first_infection_days: List[float]


def _summarise_run(sim: RegionalSimulation, seed: int,
                   experiment_name: str = "", replicate: int = 0) -> RunResult:
    """Pull headline metrics from one completed :class:`RegionalSimulation`."""
    summary = sim.regional_summary()
    peak_day = max(sim.history, key=lambda h: h["total_infectious"])
    active_days = [h["day"] for h in sim.history
                   if h["total_exposed"] + h["total_infectious"] > 0]
    config = sim.config
    network_report = summary["network_reports"][0] if summary["network_reports"] else {}
    return RunResult(
        seed=seed,
        scenario=("clustered" if config.clustered_cities else
                  config.contact_model),
        experiment_name=experiment_name,
        replicate=replicate,
        population=int(sum(config.city_sizes())),
        initial_infected=int(config.initial_infected),
        contact_model=config.contact_model,
        infection_probability=float(config.infection_probability),
        incubation_days=int(config.incubation_days),
        infectious_days=int(config.infectious_days),
        simulation_days=int(config.simulation_days),
        daily_travel_rate=float(config.daily_travel_rate),
        travel_fraction=float(config.travel_fraction),
        cluster_count=int(config.num_clusters),
        within_cluster_probability=float(
            config.within_cluster_contact_probability
            if config.within_cluster_contact_probability is not None
            else 1.0 - config.random_chance),
        average_contacts=float(np.mean([
            stats["average_contacts"] for stats in summary["contact_statistics"]
        ])),
        std_contacts=float(np.mean([
            stats["std_contacts"] for stats in summary["contact_statistics"]
        ])),
        network_node_count=float(network_report.get("node_count", float("nan"))),
        network_edge_count=float(network_report.get("edge_count", float("nan"))),
        network_mean_degree=float(network_report.get("mean_degree", float("nan"))),
        network_std_degree=float(network_report.get("std_degree", float("nan"))),
        network_clustering_coefficient=float(
            network_report.get("clustering_coefficient", float("nan"))),
        network_num_connected_components=float(
            network_report.get("num_connected_components", float("nan"))),
        network_largest_component_size=float(
            network_report.get("largest_component_size", float("nan"))),
        network_average_shortest_path_length=float(
            network_report.get("average_shortest_path_length", float("nan"))),
        network_path_length_computed=bool(
            network_report.get("path_length_computed", False)),
        average_arrival_delay=summary["average_arrival_delay"],
        cities_reached=summary["cities_reached"],
        peak_regional_infectious=int(peak_day["total_infectious"]),
        peak_regional_infectious_day=int(peak_day["day"]),
        epidemic_duration=max(active_days) if active_days else 0,
        duration_censored=bool(summary["regional_duration_censored"]),
        total_infected=int(summary["total_infected"]),
        attack_rate=summary["regional_attack_rate"],
        imported_infections=int(summary["imported_infections"]),
        first_infection_days=summary["city_first_infection_days"],
    )


def run_experiment(base_config: Config, num_runs: int = 100,
                   base_seed: int = 0, verbose: bool = False,
                   experiment_name: str = "") -> Dict:
    """Run ``num_runs`` simulations over consecutive seeds and aggregate.

    Every replicate has its own explicit seed (``base_seed + i``), applied
    via :meth:`Config.with_overrides` -- nothing here depends on hidden
    global random state, so re-running with the same ``base_config``,
    ``num_runs``, and ``base_seed`` reproduces byte-identical results (see
    ``tests/test_experiments.py::test_run_experiment_is_reproducible``).

    Args:
        base_config: The configuration to replicate (its ``random_seed`` is
            overridden per run).
        num_runs: Number of independent replications.
        base_seed: Seeds used are ``base_seed .. base_seed + num_runs - 1``.
        verbose: If True, print a progress line per run.
        experiment_name: Optional label recorded on every :class:`RunResult`
            (and thus every CSV row) for traceability when combining output
            from multiple calls.

    Returns:
        A dict with the per-run results and mean/std aggregates for the
        headline metrics.
    """
    runs: List[RunResult] = []
    for i in range(num_runs):
        seed = base_seed + i
        sim = RegionalSimulation(base_config.with_overrides(random_seed=seed))
        sim.run()
        result = _summarise_run(sim, seed, experiment_name=experiment_name, replicate=i)
        runs.append(result)
        if verbose:
            print(f"  run {i + 1}/{num_runs} (seed {seed}): "
                  f"reached {result.cities_reached}, "
                  f"delay {result.average_arrival_delay:.1f}, "
                  f"peak {result.peak_regional_infectious}")

    def agg(getter) -> Dict[str, float]:
        # Only aggregate delay over runs where the outbreak actually spread.
        values = [float(getter(r)) for r in runs if getter(r) is not None]
        values = [v for v in values if v >= 0]
        if not values:
            return {"mean": float("nan"), "std": float("nan"), "n": 0,
                    "ci95": float("nan")}
        mean, std, ci95 = mean_and_ci(values)
        return {"mean": mean, "std": std, "n": len(values), "ci95": ci95}

    return {
        "num_runs": num_runs,
        "base_seed": base_seed,
        "average_outbreak_delay": agg(lambda r: r.average_arrival_delay),
        "cities_reached": agg(lambda r: r.cities_reached),
        "peak_infections": agg(lambda r: r.peak_regional_infectious),
        "peak_infection_day": agg(lambda r: r.peak_regional_infectious_day),
        "epidemic_duration": agg(lambda r: r.epidemic_duration),
        "attack_rate": agg(lambda r: r.attack_rate),
        "imported_infections": agg(lambda r: r.imported_infections),
        "runs": runs,
    }


def write_run_results_csv(runs: List[RunResult], path: str,
                          write_footer: bool = True) -> None:
    """Write a flat list of :class:`RunResult` rows to CSV, one row per run.

    Every ``RunResult`` field becomes a column (identification, model,
    network, and outcome metadata together) -- see the field-group comments
    on :class:`RunResult` for what each one means. Used both for a single
    scenario's runs (:func:`write_experiment_csv`) and for a multi-scenario
    comparison (:func:`write_scenario_comparison_csv`), so both paths share
    exactly one column layout.

    Args:
        runs: The rows to write.
        path: Destination CSV path.
        write_footer: If True, print a one-line confirmation after writing.
    """
    fieldnames = [f.name for f in dataclasses.fields(RunResult)]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in runs:
            writer.writerow(dataclasses.asdict(r))
    if write_footer:
        print(f"Wrote {len(runs)} run(s) to {path}")


def write_experiment_csv(result: Dict, path: str) -> None:
    """Write a batch experiment's per-run results and aggregates to CSV.

    One row per run (seed + every :class:`RunResult` field), followed by a
    blank line and a compact ``metric, mean, std, ci95, n`` aggregate block,
    so the same file supports both re-analysis of individual runs and a
    quick read of the headline numbers.

    Args:
        result: The dict returned by :func:`run_experiment`.
        path: Destination CSV path.
    """
    runs: List[RunResult] = result["runs"]
    write_run_results_csv(runs, path, write_footer=False)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([])
        writer.writerow(["metric", "mean", "std", "ci95", "n"])
        for key in ("average_outbreak_delay", "cities_reached",
                    "peak_infections", "peak_infection_day",
                    "epidemic_duration", "attack_rate", "imported_infections"):
            a = result[key]
            writer.writerow([key, a["mean"], a["std"], a["ci95"], a["n"]])
    print(f"Wrote {len(runs)} experiment runs to {path}")


def print_experiment_report(result: Dict, config: Config) -> None:
    """Pretty-print an experiment aggregate to stdout."""
    def line(label: str, key: str, scale: float = 1.0, unit: str = "") -> None:
        a = result[key]
        if a["n"] == 0:
            print(f"  {label:<26} n/a")
        else:
            print(f"  {label:<26} {a['mean'] * scale:7.2f} +/- "
                  f"{a['std'] * scale:6.2f}{unit}  "
                  f"(95% CI +/-{a['ci95'] * scale:.2f}, n={a['n']})")

    print("\n" + "=" * 60)
    print(f"  EXPERIMENT: {result['num_runs']} runs "
          f"(seeds {result['base_seed']}..{result['base_seed'] + result['num_runs'] - 1})")
    print("=" * 60)
    print(f"  Cities: {config.num_cities()}   "
          f"populations: {config.city_sizes()}")
    print(f"  Estimated R0: {config.estimated_r0():.2f}")
    print("-" * 60)
    line("Avg outbreak delay (days)", "average_outbreak_delay")
    line("Cities reached", "cities_reached")
    line("Peak regional infectious", "peak_infections")
    line("Peak day", "peak_infection_day")
    line("Epidemic duration (days)", "epidemic_duration")
    line("Attack rate", "attack_rate", scale=100.0, unit="%")
    line("Imported infections", "imported_infections")
    print("=" * 60 + "\n")


#
# Scenario comparison: run several named configurations under the same
# replicate count / seed set, in one call, with no code changes per scenario.
#
def run_scenario_comparison(
    scenarios: Dict[str, Config], num_runs: int = 100, base_seed: int = 1000,
    experiment_name: str = "scenario_comparison", verbose: bool = False,
) -> Dict[str, Any]:
    """Run several named scenarios with the same replicate count and seed set.

    This is the direct answer to "run scenario A vs scenario B, 100
    replicates, starting seed 1000, without touching code": pass a
    ``{name: Config}`` mapping and everything else is handled here.

    Every scenario reuses the *same* ``base_seed .. base_seed + num_runs - 1``
    seed set (common random numbers), so a difference between scenarios in
    the aggregated results reflects the swept configuration, not seed noise
    -- the same principle :func:`run_sensitivity_analysis` already uses.
    Each replicate is fully reproducible: the same ``scenarios``, ``num_runs``,
    and ``base_seed`` always reproduce the same per-run results, because
    every random source is seeded explicitly from ``Config.random_seed``
    (see :meth:`Config.with_overrides` and :class:`RegionalSimulation`) --
    nothing here or downstream relies on hidden global random state.

    Args:
        scenarios: Mapping from a scenario label (used verbatim as the
            ``scenario`` field on every one of that scenario's
            :class:`RunResult` rows -- overriding the auto-derived label
            :func:`run_experiment` would otherwise use) to the
            :class:`~config.Config` to run for it.
        num_runs: Replicates per scenario (not hard-coded -- pass any value).
        base_seed: First seed of the shared ``base_seed .. base_seed +
            num_runs - 1`` set used by every scenario.
        experiment_name: Label recorded on every row for traceability.
        verbose: If True, print a progress line per run.

    Returns:
        Dict with ``"runs"`` (the combined flat list of every scenario's
        :class:`RunResult` rows, ready for :func:`write_run_results_csv`) and
        ``"scenarios"`` (a ``{name: run_experiment(...) result}`` dict, so
        each scenario's own aggregates from :func:`run_experiment` are still
        available individually).
    """
    per_scenario: Dict[str, Dict[str, Any]] = {}
    all_runs: List[RunResult] = []
    for name, config in scenarios.items():
        if verbose:
            print(f"Scenario '{name}': {num_runs} runs, seeds "
                  f"{base_seed}..{base_seed + num_runs - 1}")
        result = run_experiment(config, num_runs=num_runs, base_seed=base_seed,
                                verbose=verbose, experiment_name=experiment_name)
        for run in result["runs"]:
            run.scenario = name
        per_scenario[name] = result
        all_runs.extend(result["runs"])
    return {"runs": all_runs, "scenarios": per_scenario}


def write_scenario_comparison_csv(comparison: Dict[str, Any], path: str) -> None:
    """Write every scenario's runs from :func:`run_scenario_comparison` to one CSV.

    One row per (scenario, replicate) -- the full :class:`RunResult` column
    set, so summary statistics for any scenario can be recomputed later
    straight from this raw file without re-running the simulation (filter by
    the ``scenario`` column).

    Args:
        comparison: The dict returned by :func:`run_scenario_comparison`.
        path: Destination CSV path.
    """
    write_run_results_csv(comparison["runs"], path)


#
# Sensitivity analysis: sweep a parameter grid, write every run to CSV
#
def run_sensitivity_analysis(base_config: Config, param_grid: Dict[str, Sequence[Any]],
                             runs_per_combo: int = 5, base_seed: int = 0,
                             csv_path: str = None,
                             verbose: bool = False) -> List[Dict[str, Any]]:
    """Sweep a Cartesian grid of parameters and record every run's outcome.

    This answers "how does mobility influence outbreak timing and severity"
    style questions directly: sweep e.g. ``daily_travel_rate`` or
    ``travel_fraction`` (mobility), ``number_of_cities``/``population_per_city``
    (structure), or ``watts_strogatz_k``/``infection_probability``
    (connectivity/transmissibility) and read off how the outcome metrics move.

    Any :class:`~config.Config` field name is a valid grid key -- each grid
    point is applied via :meth:`Config.with_overrides`, so this requires no
    special-casing per parameter.

    Args:
        base_config: Template configuration; grid values override its fields.
        param_grid: Dict mapping a Config field name to the list of values to
            try, e.g. ``{"daily_travel_rate": [0.0, 0.05, 0.1, 0.2],
            "number_of_cities": [2, 5, 10]}``. The full Cartesian product of
            all keys is swept.
        runs_per_combo: Independent seeds run per grid point (for variance).
        base_seed: The same ``base_seed .. base_seed + runs_per_combo - 1``
            seed set is reused for every grid point (common random numbers),
            so differences between points reflect the swept parameter rather
            than seed noise.
        csv_path: If given, write one row per individual run (every swept
            parameter + every outcome metric) to this CSV path.
        verbose: If True, print a progress line per run.

    Returns:
        The list of per-run result rows (plain dicts), matching the CSV.
    """
    keys = list(param_grid.keys())
    combos = list(itertools.product(*(param_grid[k] for k in keys)))
    rows: List[Dict[str, Any]] = []

    for combo_idx, combo in enumerate(combos):
        overrides = dict(zip(keys, combo))
        combo_config = base_config.with_overrides(**overrides)
        for run_idx in range(runs_per_combo):
            seed = base_seed + run_idx
            sim = RegionalSimulation(combo_config.with_overrides(random_seed=seed))
            sim.run()
            result = _summarise_run(sim, seed)
            row: Dict[str, Any] = dict(overrides)
            row["seed"] = seed
            row["average_arrival_delay"] = result.average_arrival_delay
            row["cities_reached"] = result.cities_reached
            row["peak_regional_infectious"] = result.peak_regional_infectious
            row["peak_regional_infectious_day"] = result.peak_regional_infectious_day
            row["epidemic_duration"] = result.epidemic_duration
            row["total_infected"] = result.total_infected
            row["attack_rate"] = result.attack_rate
            row["imported_infections"] = result.imported_infections
            rows.append(row)
            if verbose:
                print(f"  combo {combo_idx + 1}/{len(combos)} "
                      f"{overrides} seed={seed}: "
                      f"reached={result.cities_reached} "
                      f"delay={result.average_arrival_delay:.1f} "
                      f"peak={result.peak_regional_infectious}")

    if csv_path:
        write_sensitivity_csv(rows, csv_path, param_keys=keys)
    return rows


def write_sensitivity_csv(rows: List[Dict[str, Any]], path: str,
                          param_keys: Sequence[str]) -> None:
    """Write sensitivity-analysis rows to CSV, swept parameters first.

    Args:
        rows: Per-run result dicts, as produced by :func:`run_sensitivity_analysis`.
        path: Destination CSV path.
        param_keys: The swept parameter names, used to order the leading columns.
    """
    if not rows:
        return
    outcome_keys = [k for k in rows[0].keys() if k not in param_keys and k != "seed"]
    fieldnames = list(param_keys) + ["seed"] + outcome_keys
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} sensitivity-analysis rows to {path}")


#
# Travel-rate comparison: a convenience wrapper over the generic sweep
#
def run_travel_rate_sweep(base_config: Config,
                          rates: Sequence[float] = (0.0, 0.05, 0.1, 0.15, 0.2),
                          runs_per_combo: int = 5, base_seed: int = 0,
                          csv_path: Optional[str] = None,
                          verbose: bool = False) -> List[Dict[str, Any]]:
    """Compare outcomes across a range of daily travel rates.

    A thin wrapper over :func:`run_sensitivity_analysis` sweeping only
    ``daily_travel_rate``, plus a printed comparison table (arrival day,
    peak infections, attack rate, epidemic duration per rate) -- the
    generic sweep already produces everything this needs, so no new
    simulation machinery is added here.

    Args:
        base_config: Template configuration; each rate overrides
            ``daily_travel_rate``.
        rates: Daily travel rates to compare.
        runs_per_combo: Independent seeds run per rate.
        base_seed: First seed of the ``base_seed .. base_seed +
            runs_per_combo - 1`` set reused for every rate.
        csv_path: If given, write every individual run to this CSV path.
        verbose: If True, print a progress line per run.

    Returns:
        The list of per-run result rows, as in :func:`run_sensitivity_analysis`.
    """
    rows = run_sensitivity_analysis(
        base_config, {"daily_travel_rate": list(rates)},
        runs_per_combo=runs_per_combo, base_seed=base_seed,
        csv_path=csv_path, verbose=verbose)

    print("\n" + "=" * 66)
    print("  TRAVEL RATE COMPARISON")
    print("=" * 66)
    print(f"  {'rate':>6} {'arrival day':>12} {'peak infect.':>13} "
          f"{'attack rate':>12} {'duration':>9}")
    for rate in rates:
        combo_rows = [r for r in rows if r["daily_travel_rate"] == rate]
        if not combo_rows:
            continue
        arrival = np.mean([r["average_arrival_delay"] for r in combo_rows
                           if r["average_arrival_delay"] >= 0] or [float("nan")])
        peak = np.mean([r["peak_regional_infectious"] for r in combo_rows])
        attack = np.mean([r["attack_rate"] for r in combo_rows])
        duration = np.mean([r["epidemic_duration"] for r in combo_rows])
        print(f"  {rate:>6.2%} {arrival:>12.1f} {peak:>13.1f} "
              f"{100 * attack:>11.1f}% {duration:>9.1f}")
    print("=" * 66 + "\n")
    return rows
