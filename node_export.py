"""Node-level per-individual CSV export.

Writes one row per person per simulated day, so a completed run can be
analysed statistically (survival curves, transmission trees, mobility
effects, ...) without rerunning the simulation. Kept separate from
``epidemic_stats.py`` (day-level aggregate counts) per the existing
disease/stats/export separation -- this module only *reads* the per-day
snapshots already recorded by :class:`~simulation.Simulation` and
:class:`~city.City`.
"""

from __future__ import annotations

import csv

from disease_model import State
from regional_simulation import RegionalSimulation
from simulation import Simulation

NODE_CSV_FIELDS = [
    "day", "person_id", "home_city", "current_city", "state",
    "days_in_state", "traveling", "infected_by", "infection_generation",
    "infection_day", "recovery_day", "contacts_today", "newly_infected",
]


def _newly_infected(state: State, days_in_state: int) -> bool:
    """A person is "newly infected" on the day they first become EXPOSED."""
    return state is State.EXPOSED and days_in_state == 0


def export_node_level_csv(regional_sim: RegionalSimulation, path: str) -> None:
    """Write one row per person per day for a completed regional run.

    Args:
        regional_sim: A :class:`~regional_simulation.RegionalSimulation` that
            has been run (its cities' ``person_frames``/``travel_status_frames``
            populated by :meth:`~city.City.record_day`).
        path: Destination CSV path.
    """
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(NODE_CSV_FIELDS)
        for city in regional_sim.cities:
            for day, (persons, locations) in enumerate(
                    zip(city.person_frames, city.travel_status_frames)):
                for person in persons:
                    current_city = locations.get(person.id, city.id)
                    writer.writerow([
                        day,
                        f"{city.id}-{person.id}",
                        city.id,
                        current_city,
                        person.state.value,
                        person.days_in_state,
                        person.id in locations,
                        person.infected_by if person.infected_by is not None else "",
                        person.infection_generation,
                        person.infection_day if person.infection_day is not None else "",
                        person.recovery_day if person.recovery_day is not None else "",
                        person.contacts_today,
                        _newly_infected(person.state, person.days_in_state),
                    ])
    print(f"Wrote node-level export to {path}")


def export_node_level_csv_single(simulation: Simulation, path: str) -> None:
    """Write one row per person per day for a completed single-city run.

    Uses the same column schema as :func:`export_node_level_csv` for a
    uniform downstream analysis pipeline; ``home_city``/``current_city`` are
    always ``0`` and ``traveling`` is always ``False`` (no travel layer in a
    single-city run).

    Args:
        simulation: A :class:`~simulation.Simulation` that has been run.
        path: Destination CSV path.
    """
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(NODE_CSV_FIELDS)
        for day, persons in enumerate(simulation.person_frames):
            for person in persons:
                writer.writerow([
                    day,
                    str(person.id),
                    0,
                    0,
                    person.state.value,
                    person.days_in_state,
                    False,
                    person.infected_by if person.infected_by is not None else "",
                    person.infection_generation,
                    person.infection_day if person.infection_day is not None else "",
                    person.recovery_day if person.recovery_day is not None else "",
                    person.contacts_today,
                    _newly_infected(person.state, person.days_in_state),
                ])
    print(f"Wrote node-level export to {path}")
