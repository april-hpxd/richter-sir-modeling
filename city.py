"""A City is an independent SEIR simulation with its own population and network.

Each City owns:
  - its own population
  - its own contact network (graph)
  - its own disease engine
  - its own RNG
  - its own history
  - its own statistics

A city advances independently day by day and can run completely by itself.
The RegionalSimulation coordinates multiple cities and adds travel between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from numpy.random import Generator

from config import Config
from disease_model import State
from engine import DiseaseEngine
from interaction import ContactModel, WattsStrogatzContactModel, WellMixedContactModel
from simulation import DailyRecord


@dataclass
class CityConfig:
    """Configuration specific to one city (a subset of the global Config).

    Each city may have slightly different settings if needed, but typically
    all cities share the same disease and contact dynamics; only population
    size and contact model parameters may vary.
    """
    population_size: int
    daily_contacts: int
    infection_probability: float
    incubation_days: int
    infectious_days: int
    contact_model_type: str
    watts_strogatz_k: int
    watts_strogatz_p: float


class City:
    """One independent city in a regional simulation.

    A City manages its own population, contact network, disease engine, and
    history. It can run completely independently or be part of a
    RegionalSimulation where it exchanges travelers with other cities.

    Attributes:
        id: Unique integer identifier for this city.
        config: Configuration for this city.
        rng: This city's independent RNG (spawned from the regional seed).
        engine: The disease engine managing this city's population.
        history: List of DailyRecord snapshots (one per day).
        state_frames: List of per-individual state snapshots (for visualization).
    """

    def __init__(self, city_id: int, config: CityConfig, rng: Generator) -> None:
        """Initialize a new city.

        Args:
            city_id: Unique identifier for this city.
            config: Configuration for this city.
            rng: The NumPy generator for this city (should be independent or
                spawned from a regional master seed).
        """
        self.id = city_id
        self.config = config
        self.rng = rng

        contact_model = self._build_contact_model()
        self.engine = DiseaseEngine(
            population_size=config.population_size,
            contact_model=contact_model,
            infection_probability=config.infection_probability,
            incubation_days=config.incubation_days,
            infectious_days=config.infectious_days,
            rng=self.rng,
        )

        self.history: List[DailyRecord] = []
        self.state_frames: List[List[State]] = []

        # Number of infections this city *imported* via returning/visiting
        # travelers (as opposed to acquiring through its own internal network).
        # Maintained by :class:`RegionalSimulation`; read-only for statistics.
        self.imported_infections: int = 0

    def _build_contact_model(self) -> ContactModel:
        """Construct the appropriate contact model for this city.

        Returns:
            A ContactModel instance (WellMixedContactModel or
            WattsStrogatzContactModel).
        """
        if self.config.contact_model_type == "well-mixed":
            return WellMixedContactModel(
                population_size=self.config.population_size,
                daily_contacts=self.config.daily_contacts,
            )
        elif self.config.contact_model_type == "watts-strogatz":
            return WattsStrogatzContactModel(
                population_size=self.config.population_size,
                k=self.config.watts_strogatz_k,
                p=self.config.watts_strogatz_p,
                rng=self.rng,
            )
        else:
            raise ValueError(
                f"Unknown contact model: {self.config.contact_model_type}"
            )

    def seed_infection(self, count: int) -> List[int]:
        """Seed ``count`` initial cases (exposed) in this city.

        Args:
            count: Number of initial cases.

        Returns:
            List of individual ids that were exposed.
        """
        return self.engine.seed_exposed(count)

    def advance_disease(self) -> Dict[str, int]:
        """Advance this city's *internal* disease dynamics by one day.

        This runs transmission and progression through the city's own contact
        network but does **not** record the day. In a regional run the record
        is deferred until after travel (see :meth:`record_day`) so that
        infections imported by travelers land in the same day's statistics and
        animation frame.

        Returns:
            The engine's daily flow dict: ``new_exposed``, ``new_infectious``,
            ``new_recovered`` (counting only internally-transmitted cases).
        """
        return self.engine.step()

    def record_day(self, new_exposed: int, new_infectious: int,
                   new_recovered: int) -> DailyRecord:
        """Capture the current engine state as a :class:`DailyRecord`.

        Also appends the per-individual state snapshot to ``state_frames`` so
        the visualization can animate this day. Call this *after* any travel
        for the day has been applied, passing the total new exposures for the
        day (internal + imported).

        Args:
            new_exposed: Total individuals newly exposed this day (internal
                transmission plus infections imported via travel).
            new_infectious: Individuals who became infectious this day.
            new_recovered: Individuals who recovered this day.

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

    def step(self) -> DailyRecord:
        """Advance and record one day, using only this city's own network.

        This is the standalone path that lets a City run completely by itself
        (no travel). A regional run instead calls :meth:`advance_disease` and
        :meth:`record_day` separately so travel can be interleaved.

        Returns:
            The DailyRecord for the day just simulated.
        """
        delta = self.advance_disease()
        return self.record_day(
            new_exposed=delta["new_exposed"],
            new_infectious=delta["new_infectious"],
            new_recovered=delta["new_recovered"],
        )

    def run(self, simulation_days: int, verbose: bool = False) -> List[DailyRecord]:
        """Run this city in isolation to completion.

        Records the day-0 baseline (if not already recorded), then steps until
        the epidemic dies out or ``simulation_days`` is reached. Demonstrates
        that a City is a self-contained simulation.

        Args:
            simulation_days: Maximum number of days to simulate.
            verbose: If True, print a one-line summary each day.

        Returns:
            The complete per-day history.
        """
        if not self.history:
            self.record_day(new_exposed=0, new_infectious=0, new_recovered=0)
        for _ in range(simulation_days):
            record = self.step()
            if verbose:
                print(f"[City {self.id}] Day {record.day:3d} | "
                      f"S={record.susceptible:3d} E={record.exposed:3d} "
                      f"I={record.infectious:3d} R={record.recovered:3d}")
            if not self.is_epidemic_active():
                break
        return self.history

    def is_epidemic_active(self) -> bool:
        """Return True if anyone is still exposed or infectious."""
        return self.engine.is_epidemic_active()

    def get_population_snapshot(self) -> Dict[int, State]:
        """Return current state of all individuals.

        Useful for identifying who can travel, who is infectious, etc.

        Returns:
            Dict mapping individual id to their current State.
        """
        return {ind.id: ind.state for ind in self.engine.individuals}

    def get_individual_state(self, individual_id: int) -> State:
        """Get the current disease state of one individual.

        Args:
            individual_id: The id of the individual.

        Returns:
            Their current State.
        """
        return self.engine.individuals[individual_id].state

    def contacts_of(self, host_id: int, rng: Generator) -> np.ndarray:
        """Return the ids a visitor would mingle with around ``host_id``.

        A traveler has no fixed node in this city, so they are attached to a
        random resident "host" and interact with that host's contacts in this
        city's own contact network (its graph neighbours for Watts-Strogatz).
        This is what makes travel spread disease through the *destination's*
        network rather than by artificial random mixing.

        Args:
            host_id: The resident whose local contacts the visitor shares.
            rng: The shared random generator (used only by stochastic models
                such as well-mixed; ignored by the static network).

        Returns:
            A 1-D array of resident ids the visitor interacts with.
        """
        return self.engine.contact_model.contacts(host_id, rng)

    @property
    def network(self):
        """The underlying contact graph, or ``None`` for the well-mixed model.

        Exposed read-only for the visualization; disease/travel logic goes
        through :meth:`contacts_of` instead of touching the graph directly.
        """
        return getattr(self.engine.contact_model, "graph", None)

    def expose(self, individual_id: int) -> bool:
        """Expose a susceptible resident (``S -> E``), e.g. from a visitor.

        No-op (returns ``False``) if the individual is not susceptible, so a
        recovered or already-infected person is never reset.

        Args:
            individual_id: The resident to expose.

        Returns:
            ``True`` if the individual was susceptible and is now exposed.
        """
        individual = self.engine.individuals[individual_id]
        if individual.state is not State.SUSCEPTIBLE:
            return False
        individual.state = State.EXPOSED
        individual.days_in_state = 0
        return True

    def set_individual_state(self, individual_id: int, state: State) -> None:
        """Forcibly set an individual's disease state.

        Used when a traveler returns from another city with an infection they
        acquired there. This is the only place external code modifies an
        individual's state.

        Args:
            individual_id: The id of the individual.
            state: The new State to assign.
        """
        self.engine.individuals[individual_id].state = state
        if state != State.SUSCEPTIBLE:
            self.engine.individuals[individual_id].days_in_state = 0

    def summary_stats(self) -> Dict[str, float]:
        """Compute summary statistics for this city.

        Returns:
            Dict with keys: population, peak_infectious, peak_infectious_day,
            peak_exposed, peak_exposed_day, total_infected, attack_rate,
            epidemic_duration_days, final_susceptible, final_recovered,
            first_infection_day (or -1 if no infection).
        """
        if not self.history:
            return {
                "population": float(self.config.population_size),
                "peak_infectious": 0.0,
                "peak_infectious_day": -1.0,
                "peak_exposed": 0.0,
                "peak_exposed_day": -1.0,
                "total_infected": 0.0,
                "attack_rate": 0.0,
                "epidemic_duration_days": 0.0,
                "final_susceptible": float(self.config.population_size),
                "final_recovered": 0.0,
                "first_infection_day": -1.0,
                "imported_infections": float(self.imported_infections),
            }

        final = self.history[-1]
        population = self.config.population_size

        # Total infected = everyone who left susceptible
        total_infected = population - final.susceptible
        attack_rate = total_infected / population if population else 0.0

        peak_inf = max(
            self.history, key=lambda r: r.infectious
        ) if self.history else self.history[0]
        peak_exp = max(
            self.history, key=lambda r: r.exposed
        ) if self.history else self.history[0]

        # Find first day with any infection (E or I)
        first_infection_day = -1
        for record in self.history:
            if record.exposed + record.infectious > 0:
                first_infection_day = record.day
                break

        # Epidemic duration = last day with active disease
        active_days = [r.day for r in self.history
                       if r.exposed + r.infectious > 0]
        epidemic_duration = max(active_days) if active_days else 0

        return {
            "population": float(population),
            "peak_infectious": float(peak_inf.infectious),
            "peak_infectious_day": float(peak_inf.day),
            "peak_exposed": float(peak_exp.exposed),
            "peak_exposed_day": float(peak_exp.day),
            "total_infected": float(total_infected),
            "attack_rate": attack_rate,
            "epidemic_duration_days": float(epidemic_duration),
            "final_susceptible": float(final.susceptible),
            "final_recovered": float(final.recovered),
            "first_infection_day": float(first_infection_day),
            "imported_infections": float(self.imported_infections),
        }
