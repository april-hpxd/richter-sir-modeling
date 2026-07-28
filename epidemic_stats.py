"""Statistics and reporting for a completed SEIR simulation.

Pure functions over a list of :class:`~simulation.DailyRecord` snapshots. They
compute the headline epidemic figures (duration, peaks, attack rate) and export
the day-by-day history to CSV. Nothing here mutates state or touches plotting,
so it can be used mid-run or after the run and shares nothing with the engine.
"""

from __future__ import annotations

import csv
from typing import Dict, List

from simulation import DailyRecord


def epidemic_duration(history: List[DailyRecord]) -> int:
    """Return the epidemic's duration in days.

    Defined as the last day on which anyone was still ``EXPOSED`` or
    ``INFECTIOUS`` (i.e. the disease was still active).

    Args:
        history: The recorded simulation history.

    Returns:
        The duration in days (``0`` if the outbreak never progressed).
    """
    active_days = [r.day for r in history if r.exposed + r.infectious > 0]
    return max(active_days) if active_days else 0


def peak(history: List[DailyRecord], attr: str) -> DailyRecord:
    """Return the :class:`DailyRecord` maximising the given count attribute.

    Args:
        history: The recorded simulation history (must be non-empty).
        attr: A record attribute name, e.g. ``"infectious"`` or ``"exposed"``.

    Returns:
        The earliest day at which ``attr`` reached its maximum.
    """
    return max(history, key=lambda r: getattr(r, attr))


def summary(history: List[DailyRecord]) -> Dict[str, float]:
    """Compute an end-of-run summary of the epidemic.

    Args:
        history: The recorded simulation history (must be non-empty).

    Returns:
        Dictionary of headline figures: population, peak infectious count and
        day, peak exposed count and day, total individuals ever infected,
        attack rate (fraction of the population ever infected), epidemic
        duration, and the final compartment counts.
    """
    final = history[-1]
    population = (final.susceptible + final.exposed
                  + final.infectious + final.recovered)
    # Everyone who ever left Susceptible was infected at some point.
    total_infected = population - final.susceptible
    attack_rate = total_infected / population if population else 0.0

    peak_inf = peak(history, "infectious")
    peak_exp = peak(history, "exposed")

    return {
        "population": float(population),
        "peak_infectious": float(peak_inf.infectious),
        "peak_infectious_day": float(peak_inf.day),
        "peak_exposed": float(peak_exp.exposed),
        "peak_exposed_day": float(peak_exp.day),
        "total_infected": float(total_infected),
        "attack_rate": attack_rate,
        "epidemic_duration_days": float(epidemic_duration(history)),
        "final_susceptible": float(final.susceptible),
        "final_recovered": float(final.recovered),
    }


def export_csv(history: List[DailyRecord], path: str) -> None:
    """Write the full day-by-day history (daily state counts) to a CSV file.

    Columns: ``day``, ``susceptible``, ``exposed``, ``infectious``,
    ``recovered``, ``new_exposed``, ``new_infectious``, ``new_recovered``.

    Args:
        history: The recorded simulation history.
        path: Destination file path.
    """
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "day", "susceptible", "exposed", "infectious", "recovered",
            "new_exposed", "new_infectious", "new_recovered",
        ])
        for r in history:
            writer.writerow([
                r.day, r.susceptible, r.exposed, r.infectious, r.recovered,
                r.new_exposed, r.new_infectious, r.new_recovered,
            ])
