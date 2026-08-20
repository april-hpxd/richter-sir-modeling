"""Presentation figures for the research question:

    "How does local social clustering -- versus a regular/random contact
    network -- affect the speed and extent of disease transmission within a
    city, and how does it propagate that effect to other cities through
    travel?"

This is a downstream analysis script, not a new simulation engine: it only
calls the existing :class:`~config.Config`, :class:`~simulation.Simulation`,
and :class:`~regional_simulation.RegionalSimulation`, the same way
``experiments.py`` does, and averages the results of many seeded
replications (common random numbers -- the same seed set is reused across
compared scenarios, so differences reflect the swept parameter, not seed
noise).

Produces four PNGs plus their backing CSVs in the working directory:
    fig1_within_city_curves.png    -- mean infectious-count curve +/- std,
                                       clustered vs regular, single city.
    within_city_curve_data.csv     -- the exact per-day mean/std behind fig1.
    fig2_within_city_peak_bar.png  -- mean peak infectious / attack rate,
                                       clustered vs regular, single city.
    within_city_run_data.csv       -- one row per seed/scenario behind fig2.
    fig3_regional_peak_bar.png     -- mean peak infectious per city,
                                       regular-vs-regular vs clustered-vs-regular.
    regional_run_data.csv          -- one row per seed/scenario/city behind fig3.
    fig4_regional_curves.png       -- mean per-city infectious curves for
                                       both regional scenarios.
    regional_curve_data.csv        -- the exact per-day mean/std behind fig4.
"""

from __future__ import annotations

import csv
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from config import Config
from regional_simulation import RegionalSimulation
from simulation import Simulation

N_SEEDS = 30
BASE_SEED = 1000
SIM_DAYS = 120

# Matched mean degree across models: random-network's default degree range
# (1..7) averages 4, so daily_contacts=4 sets the clustered model's
# intra-cluster ring degree to the same average -- this isolates "local
# clustering vs random wiring" as the one varying factor, instead of
# confounding it with "clustered people simply have more contacts."
DAILY_CONTACTS = 4

COLOR_CLUSTERED = "#e63946"   # matches STATE_COLOR[INFECTIOUS] in visualization.py
COLOR_REGULAR = "#264653"


def _padded_infectious(history, days: int) -> np.ndarray:
    """Per-day infectious count, forward-filled (with 0) out to ``days``."""
    values = [r.infectious for r in history]
    if len(values) < days + 1:
        values = values + [values[-1]] * (days + 1 - len(values))
    return np.array(values[: days + 1], dtype=float)


def _clipped_yerr(means: List[float], stds: List[float]) -> np.ndarray:
    """Asymmetric error bars whose lower whisker never crosses below zero
    (a count/rate can't be negative, so a symmetric +/-std bar is misleading
    for small, noisy means)."""
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    lower = np.minimum(means_arr, stds_arr)
    return np.vstack([lower, stds_arr])


def _peak_and_attack_rate(history, population: int) -> Tuple[int, float]:
    peak = max(r.infectious for r in history)
    final_susceptible = history[-1].susceptible
    attack_rate = (population - final_susceptible) / population
    return peak, attack_rate


#
# Study 1: within-city speed/extent, clustered vs regular
#
def run_within_city_study():
    curves = {"clustered": [], "regular": []}
    peaks = {"clustered": [], "regular": []}
    attack_rates = {"clustered": [], "regular": []}

    for label, contact_model in (("clustered", "clustered"),
                                 ("regular", "random-network")):
        for i in range(N_SEEDS):
            seed = BASE_SEED + i
            config = Config(
                population_size=50, contact_model=contact_model,
                daily_contacts=DAILY_CONTACTS,
                num_clusters=5, random_chance=0.1,
                initial_infected=1, simulation_days=SIM_DAYS,
                random_seed=seed,
            )
            sim = Simulation(config)
            sim.run()
            curves[label].append(_padded_infectious(sim.history, SIM_DAYS))
            peak, attack = _peak_and_attack_rate(sim.history, config.population_size)
            peaks[label].append(peak)
            attack_rates[label].append(attack)

    return curves, peaks, attack_rates


def write_within_city_run_csv(peaks, attack_rates) -> None:
    """One row per (seed, scenario): the raw data behind fig2's bars."""
    with open("within_city_run_data.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "scenario", "peak_infectious", "attack_rate"])
        for label in ("clustered", "regular"):
            for i in range(N_SEEDS):
                writer.writerow([BASE_SEED + i, label,
                                 peaks[label][i], attack_rates[label][i]])
    print("Wrote within_city_run_data.csv")


def write_within_city_curve_csv(curves) -> None:
    """day, clustered_mean, clustered_std, regular_mean, regular_std -- the
    exact per-day series plotted (as mean +/- std band) in fig1."""
    clustered = np.array(curves["clustered"])
    regular = np.array(curves["regular"])
    with open("within_city_curve_data.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["day", "clustered_mean", "clustered_std",
                         "regular_mean", "regular_std"])
        for day in range(SIM_DAYS + 1):
            writer.writerow([
                day, clustered[:, day].mean(), clustered[:, day].std(),
                regular[:, day].mean(), regular[:, day].std(),
            ])
    print("Wrote within_city_curve_data.csv")


def plot_within_city_curves(curves) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    days = np.arange(SIM_DAYS + 1)
    for label, color, display in (("clustered", COLOR_CLUSTERED, "Clustered"),
                                  ("regular", COLOR_REGULAR, "Regular (random-network)")):
        stack = np.array(curves[label])
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        ax.plot(days, mean, color=color, linewidth=2.4, label=display)
        ax.fill_between(days, np.maximum(0, mean - std), mean + std,
                        color=color, alpha=0.18)
    ax.set_title(f"Within-city infectious count over time\n"
                f"(mean ± 1 std over {N_SEEDS} seeds, n=50, random-chance=0.1)",
                fontsize=12, weight="bold")
    ax.set_xlabel("Day")
    ax.set_ylabel("Infectious individuals")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig1_within_city_curves.png", dpi=150)
    print("Saved fig1_within_city_curves.png")


def plot_within_city_peak_bar(peaks, attack_rates) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    labels = ["Clustered", "Regular"]
    colors = [COLOR_CLUSTERED, COLOR_REGULAR]

    peak_means = [np.mean(peaks["clustered"]), np.mean(peaks["regular"])]
    peak_stds = [np.std(peaks["clustered"]), np.std(peaks["regular"])]
    ax1.bar(labels, peak_means, yerr=_clipped_yerr(peak_means, peak_stds),
            capsize=6, color=colors)
    ax1.set_title("Peak infectious count", fontsize=12, weight="bold")
    ax1.set_ylabel("Peak infectious individuals")
    ax1.grid(alpha=0.3, axis="y")

    attack_means = [100 * np.mean(attack_rates["clustered"]),
                    100 * np.mean(attack_rates["regular"])]
    attack_stds = [100 * np.std(attack_rates["clustered"]),
                   100 * np.std(attack_rates["regular"])]
    ax2.bar(labels, attack_means, yerr=_clipped_yerr(attack_means, attack_stds),
            capsize=6, color=colors)
    ax2.set_title("Attack rate", fontsize=12, weight="bold")
    ax2.set_ylabel("% of population ever infected")
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Within-city outcome: clustered vs regular "
                f"(mean ± 1 std, n=50, {N_SEEDS} seeds each)",
                fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("fig2_within_city_peak_bar.png", dpi=150)
    print("Saved fig2_within_city_peak_bar.png")


#
# Study 2: regional propagation, clustered-source vs regular-source
#
def run_regional_study():
    scenarios = {
        "regular_regular": None,       # City 0 and City 1 both regular
        "clustered_regular": (0,),     # City 0 clustered, City 1 regular
    }
    curves = {name: {"A": [], "B": []} for name in scenarios}
    peaks = {name: {"A": [], "B": []} for name in scenarios}

    for name, clustered_cities in scenarios.items():
        for i in range(N_SEEDS):
            seed = BASE_SEED + i
            config = Config(
                city_populations=(50, 50), clustered_cities=clustered_cities,
                daily_contacts=DAILY_CONTACTS,
                num_clusters=5, random_chance=0.1,
                travel_fraction=0.3, daily_travel_rate=0.1,
                initial_infected=1, simulation_days=SIM_DAYS,
                random_seed=seed,
            )
            sim = RegionalSimulation(config)
            sim.run()
            city_a, city_b = sim.cities[0], sim.cities[1]
            curves[name]["A"].append(_padded_infectious(city_a.history, SIM_DAYS))
            curves[name]["B"].append(_padded_infectious(city_b.history, SIM_DAYS))
            peaks[name]["A"].append(max(r.infectious for r in city_a.history))
            peaks[name]["B"].append(max(r.infectious for r in city_b.history))

    return curves, peaks


def write_regional_run_csv(peaks) -> None:
    """One row per (seed, scenario, city): the raw data behind fig3's bars."""
    with open("regional_run_data.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "scenario", "city", "peak_infectious"])
        for scenario in ("regular_regular", "clustered_regular"):
            for city in ("A", "B"):
                for i in range(N_SEEDS):
                    writer.writerow([BASE_SEED + i, scenario, city,
                                     peaks[scenario][city][i]])
    print("Wrote regional_run_data.csv")


def write_regional_curve_csv(curves) -> None:
    """day, scenario, city, mean, std -- the exact per-day series plotted
    (as mean +/- std band) in fig4."""
    with open("regional_curve_data.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["day", "scenario", "city", "mean", "std"])
        for scenario in ("regular_regular", "clustered_regular"):
            for city in ("A", "B"):
                stack = np.array(curves[scenario][city])
                for day in range(SIM_DAYS + 1):
                    writer.writerow([day, scenario, city,
                                     stack[:, day].mean(), stack[:, day].std()])
    print("Wrote regional_curve_data.csv")


def plot_regional_peak_bar(peaks) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(2)  # City A, City B
    width = 0.35

    ctrl_means = [np.mean(peaks["regular_regular"]["A"]),
                 np.mean(peaks["regular_regular"]["B"])]
    ctrl_stds = [np.std(peaks["regular_regular"]["A"]),
                np.std(peaks["regular_regular"]["B"])]
    exp_means = [np.mean(peaks["clustered_regular"]["A"]),
                np.mean(peaks["clustered_regular"]["B"])]
    exp_stds = [np.std(peaks["clustered_regular"]["A"]),
               np.std(peaks["clustered_regular"]["B"])]

    ax.bar(x - width / 2, ctrl_means, width, yerr=_clipped_yerr(ctrl_means, ctrl_stds),
           capsize=6, color=COLOR_REGULAR, label="Regular ↔ Regular (control)")
    ax.bar(x + width / 2, exp_means, width, yerr=_clipped_yerr(exp_means, exp_stds),
           capsize=6, color=COLOR_CLUSTERED, label="Clustered A ↔ Regular B")

    ax.set_xticks(x)
    ax.set_xticklabels(["City A (source)", "City B (neighbour)"])
    ax.set_ylabel("Peak infectious individuals")
    ax.set_title(f"Regional peak infections: clustered-source vs regular-source\n"
                f"(mean ± 1 std, n=50 per city, {N_SEEDS} seeds each)",
                fontsize=12, weight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("fig3_regional_peak_bar.png", dpi=150)
    print("Saved fig3_regional_peak_bar.png")


def plot_regional_curves(curves) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    days = np.arange(SIM_DAYS + 1)

    panel_specs = [
        ("regular_regular", "Control: City A regular ↔ City B regular", axes[0]),
        ("clustered_regular", "City A clustered ↔ City B regular", axes[1]),
    ]
    for name, title, ax in panel_specs:
        for city_key, color, display in (("A", "#f4a261", "City A"),
                                         ("B", "#457b9d", "City B")):
            stack = np.array(curves[name][city_key])
            mean = stack.mean(axis=0)
            std = stack.std(axis=0)
            ax.plot(days, mean, color=color, linewidth=2.4, label=display)
            ax.fill_between(days, np.maximum(0, mean - std), mean + std,
                            color=color, alpha=0.18)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel("Day")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Infectious individuals")
    fig.suptitle(f"Regional propagation: mean infectious curves per city "
                f"(± 1 std, {N_SEEDS} seeds each)", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("fig4_regional_curves.png", dpi=150)
    print("Saved fig4_regional_curves.png")


def main() -> None:
    print(f"Running within-city study ({N_SEEDS} seeds x 2 scenarios)...")
    curves1, peaks1, attack1 = run_within_city_study()
    plot_within_city_curves(curves1)
    write_within_city_curve_csv(curves1)
    plot_within_city_peak_bar(peaks1, attack1)
    write_within_city_run_csv(peaks1, attack1)

    print(f"Running regional study ({N_SEEDS} seeds x 2 scenarios)...")
    curves2, peaks2 = run_regional_study()
    plot_regional_peak_bar(peaks2)
    write_regional_run_csv(peaks2)
    plot_regional_curves(curves2)
    write_regional_curve_csv(curves2)

    print("\nSummary (mean values):")
    print(f"  Within-city  clustered: peak={np.mean(peaks1['clustered']):.1f}  "
          f"attack_rate={100*np.mean(attack1['clustered']):.1f}%")
    print(f"  Within-city  regular:   peak={np.mean(peaks1['regular']):.1f}  "
          f"attack_rate={100*np.mean(attack1['regular']):.1f}%")
    print(f"  Regional control  City A peak={np.mean(peaks2['regular_regular']['A']):.1f}  "
          f"City B peak={np.mean(peaks2['regular_regular']['B']):.1f}")
    print(f"  Regional exp.     City A peak={np.mean(peaks2['clustered_regular']['A']):.1f}  "
          f"City B peak={np.mean(peaks2['clustered_regular']['B']):.1f}")


if __name__ == "__main__":
    main()
