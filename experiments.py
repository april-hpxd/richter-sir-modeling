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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    vaccination_rate: float

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
        vaccination_rate=float(config.vaccination_rate),
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


#
# Vaccination-coverage sweep: dose-response experiment over
# Config.vaccination_rate, with major-outbreak / invasion-conditioned
# analysis. Built entirely on the existing per-run machinery above
# (_summarise_run / RunResult) -- no new simulation behaviour, no change to
# how a single run is executed.
#
DEFAULT_MAJOR_OUTBREAK_THRESHOLD = 20
DEFAULT_VACCINATION_COVERAGE_RATES: Tuple[float, ...] = tuple(
    round(0.05 * i, 2) for i in range(13))  # 0.00, 0.05, ..., 0.60


def is_major_outbreak(run: RunResult,
                      threshold: int = DEFAULT_MAJOR_OUTBREAK_THRESHOLD) -> bool:
    """Whether a run's total infections meet the major-outbreak threshold.

    Definition is deliberately a fixed, documented threshold on
    ``total_infected`` (default ``20``, matching the analysis the mentor's
    100-run pilot already reported) -- never changed silently. Pass
    ``threshold`` explicitly to use a different cutoff.
    """
    return run.total_infected >= threshold


def is_successful_invasion(run: RunResult) -> bool:
    """Whether the outbreak spread beyond the seed city (city 0) at all.

    ``cities_reached`` counts city 0 itself, so ``> 1`` means at least one
    other city was infected. In the project's usual 2-city setup this is
    exactly "City B reached" / the mentor's "City-2 invasion"; the same
    definition generalises to "at least one other city reached" for more
    than two cities. A non-invasion run is a legitimate outcome (the seeded
    cases burned out locally), not missing data -- see
    :func:`summarize_vaccination_coverage`.
    """
    return run.cities_reached > 1


def run_vaccination_coverage_sweep(
    base_config: Config,
    vaccination_rates: Sequence[float] = DEFAULT_VACCINATION_COVERAGE_RATES,
    num_runs: int = 500,
    base_seed: int = 0,
    verbose: bool = False,
    experiment_name: str = "vaccination_coverage",
) -> List[RunResult]:
    """Run a dose-response experiment over ``Config.vaccination_rate``.

    Each rate reuses the *same* ``base_seed .. base_seed + num_runs - 1``
    seed set (common random numbers, exactly as :func:`run_scenario_comparison`
    and :func:`run_sensitivity_analysis` already do), so a difference between
    rates reflects the swept vaccination coverage, not seed noise. Every
    other field of ``base_config`` -- population, contact model, travel,
    disease parameters -- is held fixed; only ``vaccination_rate`` varies.

    Reproducibility: the same ``base_config``, ``vaccination_rates``,
    ``num_runs``, and ``base_seed`` always reproduce the same per-run
    results (:class:`RunResult` rows), because every random draw is seeded
    explicitly from ``Config.random_seed`` -- see
    :func:`run_experiment`'s docstring for the same contract. A rate of
    ``0.0`` runs through exactly the same no-vaccination code path as any
    other ``Config`` with ``vaccination_rate=0.0`` (vaccination draws no
    extra randomness when disabled -- see ``vaccination.py`` and
    ``Simulation.__init__``/``City.__init__``), so it reproduces the
    pre-vaccination behaviour exactly.

    Args:
        base_config: Template configuration; each point overrides only
            ``vaccination_rate``.
        vaccination_rates: Coverage levels to compare. Defaults to
            ``0.00, 0.05, ..., 0.60`` (13 points).
        num_runs: Replicates per rate (the mentor's pilot used 100; the next
            round of replication calls for approximately 500-1000 -- pass
            any value, e.g. a small ``5`` for a quick smoke test before a
            full run).
        base_seed: First seed of the shared ``base_seed .. base_seed +
            num_runs - 1`` set reused by every rate.
        verbose: If True, print a progress line per run.
        experiment_name: Label recorded on every row.

    Returns:
        The flat list of every rate's :class:`RunResult` rows (``len() ==
        len(vaccination_rates) * num_runs``), each with its ``scenario``
        field set to ``"vaccination_rate=<rate>"`` for easy filtering, ready
        for :func:`write_run_results_csv` or
        :func:`summarize_vaccination_coverage`.
    """
    all_runs: List[RunResult] = []
    for rate in vaccination_rates:
        rate_config = base_config.with_overrides(vaccination_rate=float(rate))
        if verbose:
            print(f"Vaccination rate {rate:.0%}: {num_runs} runs, seeds "
                  f"{base_seed}..{base_seed + num_runs - 1}")
        for i in range(num_runs):
            seed = base_seed + i
            sim = RegionalSimulation(rate_config.with_overrides(random_seed=seed))
            sim.run()
            run = _summarise_run(sim, seed, experiment_name=experiment_name,
                                 replicate=i)
            run.scenario = f"vaccination_rate={rate:g}"
            all_runs.append(run)
            if verbose:
                print(f"  rate={rate:.0%} run {i + 1}/{num_runs} (seed {seed}): "
                      f"total_infected={run.total_infected}, "
                      f"invaded={is_successful_invasion(run)}")
    return all_runs


def _clip01(value: float) -> float:
    """Clamp a proportion's normal-approximation CI bound into [0, 1]."""
    if value != value:  # NaN
        return value
    return max(0.0, min(1.0, value))


def summarize_vaccination_coverage(
    runs: List[RunResult],
    major_outbreak_threshold: int = DEFAULT_MAJOR_OUTBREAK_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Aggregate :func:`run_vaccination_coverage_sweep` output, one row per rate.

    Every statistic reuses :func:`analysis.mean_and_ci` (the project's one
    existing mean/CI implementation -- a normal-approximation 95% CI, applied
    here to continuous outcomes and, for the two proportions
    (``invasion_probability``, major-outbreak rate), to the 0/1 indicator
    values) rather than introducing a second statistical method.

    Invasion/arrival-delay handling (a run where the outbreak never spread
    past the seed city is a real outcome, not missing data -- see
    :func:`is_successful_invasion`):
      - ``invasion_probability`` and its CI are computed over **all** ``n``
        runs (denominator = every run, non-invasions count as ``0``).
      - ``mean_arrival_delay_successful`` is computed **only** over runs
        that actually invaded (denominator = ``successful_invasions``,
        reported explicitly); a run with no invasion contributes nothing to
        it, rather than being coerced into a ``-1`` or ``0`` delay.

    Major-outbreak handling (fixed, documented threshold on
    ``total_infected``, see :func:`is_major_outbreak` -- never silently
    changed): ``major_outbreak_count``/``major_outbreak_percent`` use **all**
    ``n`` runs as the denominator; every ``major_outbreak_*`` statistic is
    computed **only** over the qualifying subset (its own count is in
    ``major_outbreak_count``).

    Censoring (see ``epidemic_stats.is_duration_censored`` /
    ``RunResult.duration_censored``, unchanged from Milestone 3): this
    function does not average censored and uncensored durations into one
    unlabelled number without saying so -- it reports ``n_censored`` and
    ``censored_fraction`` alongside ``mean_duration`` so a censored-heavy
    rate's duration figure can be read with that caveat in mind, rather than
    silently treating a simulation-horizon cutoff as a true extinction time.

    Args:
        runs: Flat list of :class:`RunResult` rows, e.g. from
            :func:`run_vaccination_coverage_sweep`.
        major_outbreak_threshold: ``total_infected`` cutoff for a "major
            outbreak" (default ``20``).

    Returns:
        One dict per distinct ``vaccination_rate`` present in ``runs``,
        sorted ascending, with the columns documented in
        :func:`write_vaccination_coverage_summary_csv`.
    """
    rates = sorted({run.vaccination_rate for run in runs})
    summary_rows: List[Dict[str, Any]] = []
    for rate in rates:
        rate_runs = [r for r in runs if r.vaccination_rate == rate]
        n = len(rate_runs)

        attack_rates = [r.attack_rate for r in rate_runs]
        peaks = [r.peak_regional_infectious for r in rate_runs]
        durations = [r.epidemic_duration for r in rate_runs]
        imported = [r.imported_infections for r in rate_runs]
        invaded = [1.0 if is_successful_invasion(r) else 0.0 for r in rate_runs]
        major = [is_major_outbreak(r, major_outbreak_threshold) for r in rate_runs]
        n_censored = sum(1 for r in rate_runs if r.duration_censored)

        attack_mean, _, attack_ci = mean_and_ci(attack_rates)
        peak_mean, _, peak_ci = mean_and_ci(peaks)
        duration_mean, _, duration_ci = mean_and_ci(durations)
        invasion_mean, _, invasion_ci = mean_and_ci(invaded)
        imported_mean, _, _ = mean_and_ci(imported)

        successful = [r for r in rate_runs if is_successful_invasion(r)]
        arrival_delays_successful = [r.average_arrival_delay for r in successful]

        major_runs = [r for r in rate_runs if is_major_outbreak(r, major_outbreak_threshold)]
        major_attack_rates = sorted(r.attack_rate for r in major_runs)
        major_invaded = [1.0 if is_successful_invasion(r) else 0.0 for r in major_runs]

        summary_rows.append({
            "vaccination_rate": rate,
            "n_runs": n,
            "mean_attack_rate": attack_mean,
            "median_attack_rate": float(np.median(attack_rates)) if attack_rates else float("nan"),
            "attack_rate_ci_low": _clip01(attack_mean - attack_ci),
            "attack_rate_ci_high": _clip01(attack_mean + attack_ci),
            "mean_peak_infectious": peak_mean,
            "peak_ci_low": peak_mean - peak_ci,
            "peak_ci_high": peak_mean + peak_ci,
            "mean_duration": duration_mean,
            "duration_ci_low": duration_mean - duration_ci,
            "duration_ci_high": duration_mean + duration_ci,
            "n_censored": n_censored,
            "censored_fraction": n_censored / n if n else float("nan"),
            "invasion_probability": invasion_mean,
            "invasion_ci_low": _clip01(invasion_mean - invasion_ci),
            "invasion_ci_high": _clip01(invasion_mean + invasion_ci),
            "successful_invasions": len(successful),
            "mean_arrival_delay_successful": (
                float(np.mean(arrival_delays_successful))
                if arrival_delays_successful else float("nan")),
            "mean_imported_infections": imported_mean,
            "major_outbreak_count": len(major_runs),
            "major_outbreak_percent": 100.0 * len(major_runs) / n if n else float("nan"),
            "major_outbreak_median_attack_rate": (
                float(np.median(major_attack_rates)) if major_attack_rates
                else float("nan")),
            # -- Major-outbreaks-only breakdown (denominator: major_outbreak_count) --
            "major_outbreak_mean_peak_infectious": (
                float(np.mean([r.peak_regional_infectious for r in major_runs]))
                if major_runs else float("nan")),
            "major_outbreak_mean_duration": (
                float(np.mean([r.epidemic_duration for r in major_runs]))
                if major_runs else float("nan")),
            "major_outbreak_invasion_probability": (
                float(np.mean(major_invaded)) if major_invaded else float("nan")),
            "major_outbreak_mean_imported_infections": (
                float(np.mean([r.imported_infections for r in major_runs]))
                if major_runs else float("nan")),
        })
    return summary_rows


VACCINATION_COVERAGE_SUMMARY_FIELDS = [
    "vaccination_rate", "n_runs",
    "mean_attack_rate", "median_attack_rate",
    "attack_rate_ci_low", "attack_rate_ci_high",
    "mean_peak_infectious", "peak_ci_low", "peak_ci_high",
    "mean_duration", "duration_ci_low", "duration_ci_high",
    "invasion_probability", "invasion_ci_low", "invasion_ci_high",
    "successful_invasions", "mean_arrival_delay_successful",
    "mean_imported_infections",
    "major_outbreak_count", "major_outbreak_percent",
    "major_outbreak_median_attack_rate",
    # Extra detail beyond the suggested column list: censoring visibility
    # (Step 6) and the major-outbreaks-only breakdown (Step 4.B).
    "n_censored", "censored_fraction",
    "major_outbreak_mean_peak_infectious", "major_outbreak_mean_duration",
    "major_outbreak_invasion_probability",
    "major_outbreak_mean_imported_infections",
]


def write_vaccination_coverage_runs_csv(runs: List[RunResult], path: str) -> None:
    """Write every replicate of a vaccination-coverage sweep, one row each.

    Thin wrapper over :func:`write_run_results_csv` (the canonical
    per-:class:`RunResult` writer) -- no separate column layout is
    introduced for this experiment's raw output.
    """
    write_run_results_csv(runs, path)


def write_vaccination_coverage_summary_csv(
    summary_rows: List[Dict[str, Any]], path: str) -> None:
    """Write one row per vaccination rate (see :func:`summarize_vaccination_coverage`).

    Column order matches the dose-response-friendly layout requested for
    this experiment, plus a few extra columns (prefixed ``major_outbreak_``,
    plus ``n_censored``/``censored_fraction``) documented in
    :data:`VACCINATION_COVERAGE_SUMMARY_FIELDS`.
    """
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=VACCINATION_COVERAGE_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {len(summary_rows)} vaccination-coverage summary row(s) to {path}")


def print_vaccination_coverage_report(summary_rows: List[Dict[str, Any]],
                                      major_outbreak_threshold: int =
                                      DEFAULT_MAJOR_OUTBREAK_THRESHOLD) -> None:
    """Print the dose-response summary: ALL RUNS, then MAJOR OUTBREAKS ONLY.

    Every percentage is printed with its explicit numerator/denominator
    (e.g. ``"City-2 invasion: 30.0% (30/100 runs)"``), per the project's
    requirement that no proportion be reported without saying what it's a
    fraction of.
    """
    print("\n" + "=" * 78)
    print("  VACCINATION COVERAGE EXPERIMENT -- ALL RUNS")
    print(f"  (major outbreak defined as total_infected >= {major_outbreak_threshold})")
    print("=" * 78)
    header = (f"  {'rate':>6} {'n':>5} {'attack%':>18} {'peak':>14} "
              f"{'duration':>14} {'invasion':>22} {'major%':>8}")
    print(header)
    for row in summary_rows:
        n = row["n_runs"]
        print(
            f"  {row['vaccination_rate']:>6.0%} {n:>5} "
            f"{100 * row['mean_attack_rate']:>6.1f} "
            f"[{100 * row['attack_rate_ci_low']:.1f},{100 * row['attack_rate_ci_high']:.1f}]  "
            f"{row['mean_peak_infectious']:>6.1f}       "
            f"{row['mean_duration']:>6.1f}       "
            f"{100 * row['invasion_probability']:>5.1f}% "
            f"({int(round(row['invasion_probability'] * n))}/{n})   "
            f"{row['major_outbreak_percent']:>6.1f}%"
        )
        if row["n_censored"]:
            print(f"    note: {row['n_censored']}/{n} runs were still active at the "
                  "simulation horizon (duration is a lower bound for those runs).")
        if row["successful_invasions"]:
            print(f"    arrival delay among successful invasions: "
                  f"{row['mean_arrival_delay_successful']:.1f} days "
                  f"(n={row['successful_invasions']}/{n})")
        else:
            print(f"    arrival delay among successful invasions: n/a (0/{n} invaded)")

    print("\n" + "=" * 78)
    print("  MAJOR OUTBREAKS ONLY "
          f"(total_infected >= {major_outbreak_threshold})")
    print("=" * 78)
    for row in summary_rows:
        n = row["n_runs"]
        qualifying = row["major_outbreak_count"]
        print(f"  rate={row['vaccination_rate']:.0%}: "
              f"{qualifying}/{n} runs qualify "
              f"({row['major_outbreak_percent']:.1f}%)")
        if qualifying == 0:
            print("    no qualifying runs at this rate.")
            continue
        print(f"    median attack rate:        "
              f"{100 * row['major_outbreak_median_attack_rate']:.1f}%")
        print(f"    mean peak infectious:      "
              f"{row['major_outbreak_mean_peak_infectious']:.1f}")
        print(f"    mean epidemic duration:    "
              f"{row['major_outbreak_mean_duration']:.1f} days")
        print(f"    invasion probability:      "
              f"{100 * row['major_outbreak_invasion_probability']:.1f}% "
              f"({int(round(row['major_outbreak_invasion_probability'] * qualifying))}/{qualifying})")
        print(f"    mean imported infections:  "
              f"{row['major_outbreak_mean_imported_infections']:.2f}")
    print("=" * 78 + "\n")
