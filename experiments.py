"""Repeated-simulation experiment framework for research figures.

Runs the same regional configuration many times under different random seeds
and aggregates the outcomes (mean and standard deviation). 

The framework only *reads* completed simulations -- it adds no disease, travel,
or visualization behaviour of its own.
"""

from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from config import Config
from regional_simulation import RegionalSimulation


@dataclass
class RunResult:
    """Metrics extracted from a single completed regional run."""

    seed: int
    average_arrival_delay: float
    cities_reached: int
    peak_regional_infectious: int
    peak_regional_infectious_day: int
    epidemic_duration: int
    total_infected: int
    attack_rate: float
    imported_infections: int
    first_infection_days: List[float]


def _summarise_run(sim: RegionalSimulation, seed: int) -> RunResult:
    """Pull headline metrics from one completed :class:`RegionalSimulation`."""
    summary = sim.regional_summary()
    peak_day = max(sim.history, key=lambda h: h["total_infectious"])
    active_days = [h["day"] for h in sim.history
                   if h["total_exposed"] + h["total_infectious"] > 0]
    return RunResult(
        seed=seed,
        average_arrival_delay=summary["average_arrival_delay"],
        cities_reached=summary["cities_reached"],
        peak_regional_infectious=int(peak_day["total_infectious"]),
        peak_regional_infectious_day=int(peak_day["day"]),
        epidemic_duration=max(active_days) if active_days else 0,
        total_infected=int(summary["total_infected"]),
        attack_rate=summary["regional_attack_rate"],
        imported_infections=int(summary["imported_infections"]),
        first_infection_days=summary["city_first_infection_days"],
    )


def run_experiment(base_config: Config, num_runs: int = 100,
                   base_seed: int = 0, verbose: bool = False) -> Dict:
    """Run ``num_runs`` simulations over consecutive seeds and aggregate.

    Args:
        base_config: The configuration to replicate (its ``random_seed`` is
            overridden per run).
        num_runs: Number of independent replications.
        base_seed: Seeds used are ``base_seed .. base_seed + num_runs - 1``.
        verbose: If True, print a progress line per run.

    Returns:
        A dict with the per-run results and mean/std aggregates for the
        headline metrics.
    """
    runs: List[RunResult] = []
    for i in range(num_runs):
        seed = base_seed + i
        sim = RegionalSimulation(base_config.with_overrides(random_seed=seed))
        sim.run()
        result = _summarise_run(sim, seed)
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
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        arr = np.array(values, dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)),
                "n": len(values)}

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


def print_experiment_report(result: Dict, config: Config) -> None:
    """Pretty-print an experiment aggregate to stdout."""
    def line(label: str, key: str, scale: float = 1.0, unit: str = "") -> None:
        a = result[key]
        if a["n"] == 0:
            print(f"  {label:<26} n/a")
        else:
            print(f"  {label:<26} {a['mean'] * scale:7.2f} +/- "
                  f"{a['std'] * scale:6.2f}{unit}  (n={a['n']})")

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
