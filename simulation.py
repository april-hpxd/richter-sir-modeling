"""Simulation orchestration and day-by-day history recording.

:class:`Simulation` wires the configuration, the interaction layer, and the
disease engine together, runs the outbreak, and records a :class:`DailyRecord`
for every day. It owns the single NumPy :class:`~numpy.random.Generator` that
drives *all* randomness in the run (seeding, contacts, transmission), which is
what makes a given :class:`~config.Config` perfectly reproducible.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from config import Config
from disease_model import State
from engine import DiseaseEngine
from interaction import (
    RandomNetworkContactModel,
    WattsStrogatzContactModel,
    WellMixedContactModel,
)


@dataclass
class DailyRecord:
    """Snapshot of the epidemic on a single day.

    Attributes:
        day: The day index (``0`` is the seeded initial state, before any step).
        susceptible: Number of ``SUSCEPTIBLE`` individuals.
        exposed: Number of ``EXPOSED`` (incubating) individuals.
        infectious: Number of ``INFECTIOUS`` individuals.
        recovered: Number of ``RECOVERED`` individuals.
        new_exposed: Individuals newly exposed on this day.
        new_infectious: Individuals who became infectious on this day.
        new_recovered: Individuals who recovered on this day.
    """

    day: int
    susceptible: int
    exposed: int
    infectious: int
    recovered: int
    new_exposed: int
    new_infectious: int
    new_recovered: int


class Simulation:
    """Run and record a single-city SEIR epidemic in a well-mixed population.

    Attributes:
        config: The configuration governing this run.
        rng: The single shared NumPy generator for all randomness.
        engine: The :class:`~engine.DiseaseEngine` advancing the disease.
        history: One :class:`DailyRecord` per day, including day 0.
        state_frames: One per-individual state snapshot per day, aligned with
            ``history``; consumed by the visualization to animate colour
            changes. Kept separate from :class:`DailyRecord` so the record
            stays a clean numeric summary.
    """

    def __init__(self, config: Config) -> None:
        """Build the interaction layer and engine, then seed initial cases.

        Args:
            config: Simulation parameters.
        """
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

        contact_model = self._build_contact_model(config)
        self.engine = DiseaseEngine(
            population_size=config.population_size,
            contact_model=contact_model,
            infection_probability=config.infection_probability,
            incubation_days=config.incubation_days,
            infectious_days=config.infectious_days,
            rng=self.rng,
        )

        # Seed patient zeros as EXPOSED and record the day-0 baseline.
        self.engine.seed_exposed(config.initial_infected)
        self.history: List[DailyRecord] = []
        self.state_frames: List[List[State]] = []
        self._record(new_exposed=config.initial_infected,
                     new_infectious=0, new_recovered=0)

    def _build_contact_model(self, config: Config):
        """Construct the appropriate contact model.

        Args:
            config: The simulation configuration.

        Returns:
            A ContactModel instance.
        """
        if config.contact_model == "random-network":
            return RandomNetworkContactModel(
                population_size=config.population_size,
                min_degree=config.random_degree_min,
                max_degree=config.random_degree_max,
                rng=self.rng,
            )
        if config.contact_model == "well-mixed":
            return WellMixedContactModel(
                population_size=config.population_size,
                daily_contacts=config.daily_contacts,
            )
        elif config.contact_model == "watts-strogatz":
            return WattsStrogatzContactModel(
                population_size=config.population_size,
                k=config.watts_strogatz_k,
                p=config.watts_strogatz_p,
                rng=self.rng,
            )
        else:
            raise ValueError(f"Unknown contact model: {config.contact_model}")

    # ------------------------------------------------------------------
    # Driving the simulation
    # ------------------------------------------------------------------
    def step(self) -> DailyRecord:
        """Advance the epidemic by one day and record it.

        Returns:
            The :class:`DailyRecord` for the day just simulated.
        """
        delta = self.engine.step()
        return self._record(
            new_exposed=delta["new_exposed"],
            new_infectious=delta["new_infectious"],
            new_recovered=delta["new_recovered"],
        )

    def run(self, verbose: bool = False) -> List[DailyRecord]:
        """Run the epidemic to completion and return the full history.

        Stops once no ``EXPOSED`` or ``INFECTIOUS`` individuals remain (the
        state can no longer change) or after ``simulation_days`` days.

        Args:
            verbose: If ``True``, print a one-line summary of each day.

        Returns:
            The complete history, including the day-0 baseline.
        """
        for _ in range(self.config.simulation_days):
            record = self.step()
            if verbose:
                print(
                    f"Day {record.day:3d} | "
                    f"S={record.susceptible:3d} "
                    f"E={record.exposed:3d} "
                    f"I={record.infectious:3d} "
                    f"R={record.recovered:3d} "
                    f"(+{record.new_exposed} exposed)"
                )
            if not self.engine.is_epidemic_active():
                break
        return self.history

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _record(self, new_exposed: int, new_infectious: int,
                new_recovered: int) -> DailyRecord:
        """Capture the current engine state as a :class:`DailyRecord`.

        Also appends the per-individual state snapshot to ``state_frames``.

        Args:
            new_exposed: New exposures attributed to this day.
            new_infectious: New infectious conversions this day.
            new_recovered: New recoveries this day.

        Returns:
            The appended :class:`DailyRecord`.
        """
        counts = self.engine.counts()
        record = DailyRecord(
            day=self.engine.day,
            susceptible=counts["S"],
            exposed=counts["E"],
            infectious=counts["I"],
            recovered=counts["R"],
            new_exposed=new_exposed,
            new_infectious=new_infectious,
            new_recovered=new_recovered,
        )
        self.history.append(record)
        self.state_frames.append(self.engine.states())
        return record
