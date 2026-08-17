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
from disease_model import PersonSnapshot, State
from engine import DiseaseEngine
from interaction import (
    ClusteredContactModel,
    ContactModel,
    RandomNetworkContactModel,
    WattsStrogatzContactModel,
    WellMixedContactModel,
)
from simulation import DailyRecord


@dataclass
class DiseaseToken:
    """A detached snapshot of one person's disease state.

    Carried by the travel layer while a resident is away, so their disease can
    progress and change city-to-city without the engine needing to track where
    they are. Mutable: the travel layer updates it in place each day.
    """

    state: State
    days_in_state: int
    infected_by: Optional[str] = None
    infection_generation: int = 0
    infection_day: Optional[int] = None
    recovery_day: Optional[int] = None


@dataclass
class VisitDayResult:
    """Outcome of one day a visitor spends in a host city.

    Attributes:
        infected_resident_ids: Residents the (infectious) visitor exposed.
        acquired_from: Id of the infectious resident that exposed the visitor
            this day, or ``None`` if the visitor was not newly infected.
    """

    infected_resident_ids: List[int]
    acquired_from: Optional[int]


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
    random_degree_min: int
    random_degree_max: int
    num_clusters: int = 4
    random_chance: float = 0.1
    behavioral_response_factor: Optional[float] = None
    isolation_contact_multiplier: float = 1.0


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
            id_prefix=f"{city_id}-",
            behavioral_response_factor=config.behavioral_response_factor,
            isolation_contact_multiplier=config.isolation_contact_multiplier,
        )

        self.history: List[DailyRecord] = []
        self.state_frames: List[List[State]] = []
        # Per-day list of PersonSnapshot, one per resident; feeds the
        # node-level CSV export (see node_export.py).
        self.person_frames: List[List[PersonSnapshot]] = []
        # Per-day sets of resident ids away travelling and their destination
        # city id, populated by RegionalSimulation before record_day; used by
        # the node export and the visualization's traveler highlight.
        self.travel_status_frames: List[Dict[int, int]] = []
        # Per-day list of (source_id, target_id) in-city transmissions,
        # populated by record_day from the engine's step() result; used by
        # the visualization's transmission-flash effect.
        self.transmission_frames: List[List] = []

        # Number of infections this city *imported* via returning/visiting
        # travelers (as opposed to acquiring through its own internal network).
        # Maintained by :class:`RegionalSimulation`; read-only for statistics.
        self.imported_infections: int = 0
        self.isolated: bool = False

    def _build_contact_model(self) -> ContactModel:
        """Construct the appropriate contact model for this city.

        Returns:
            A ContactModel instance (WellMixedContactModel or
            WattsStrogatzContactModel).
        """
        if self.config.contact_model_type == "random-network":
            return RandomNetworkContactModel(
                population_size=self.config.population_size,
                min_degree=self.config.random_degree_min,
                max_degree=self.config.random_degree_max,
                rng=self.rng,
            )
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
        elif self.config.contact_model_type == "clustered":
            return ClusteredContactModel(
                population_size=self.config.population_size,
                num_clusters=self.config.num_clusters,
                random_chance=self.config.random_chance,
                daily_contacts=self.config.daily_contacts,
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
                   new_recovered: int, transmissions: Optional[List] = None,
                   traveler_locations: Optional[Dict[int, int]] = None
                   ) -> DailyRecord:
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
            transmissions: This day's in-city ``(source_id, target_id)``
                exposure pairs (from the engine's ``step()`` result), used by
                the visualization's transmission-flash effect.
            traveler_locations: Map of this city's resident ids currently
                travelling to their destination city id, used by the node
                export and the visualization's traveler highlight.

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
        self.transmission_frames.append(list(transmissions or []))
        self.travel_status_frames.append(dict(traveler_locations or {}))
        self.person_frames.append(self._snapshot_persons())
        return record

    def _snapshot_persons(self) -> List[PersonSnapshot]:
        """Build this day's :class:`PersonSnapshot` list for the node export."""
        counts = self.engine.last_contact_counts
        cluster_of = getattr(self.engine.contact_model, "cluster_of", None)
        snapshots = []
        for ind in self.engine.individuals:
            if not ind.present:
                # Away travelling: today's contacts happened in the
                # destination city's network, not this one, so this city's
                # nominal degree would be misleading here.
                contacts_today = 0
            else:
                contacts_today = counts.get(ind.id, self.engine.nominal_contacts(ind.id))
            snapshots.append(PersonSnapshot(
                id=ind.id,
                state=ind.state,
                days_in_state=ind.days_in_state,
                infected_by=ind.infected_by,
                infection_generation=ind.infection_generation,
                infection_day=ind.infection_day,
                recovery_day=ind.recovery_day,
                contacts_today=contacts_today,
                cluster_id=(int(cluster_of[ind.id]) if cluster_of is not None else None),
            ))
        return snapshots

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
            transmissions=delta.get("transmissions"),
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

    def expose(self, individual_id: int, infected_by: Optional[str] = None,
              infection_generation: int = 0,
              infection_day: Optional[int] = None) -> bool:
        """Expose a susceptible resident (``S -> E``), e.g. from a visitor.

        No-op (returns ``False``) if the individual is not susceptible, so a
        recovered or already-infected person is never reset.

        Args:
            individual_id: The resident to expose.
            infected_by: Global id of the source (see
                :attr:`disease_model.Individual.infected_by`).
            infection_generation: The new transmission generation.
            infection_day: The day of exposure.

        Returns:
            ``True`` if the individual was susceptible and is now exposed.
        """
        individual = self.engine.individuals[individual_id]
        if individual.state is not State.SUSCEPTIBLE:
            return False
        individual.state = State.EXPOSED
        individual.days_in_state = 0
        individual.infected_by = infected_by
        individual.infection_generation = infection_generation
        individual.infection_day = infection_day
        return True

    def set_isolated(self, isolated: bool) -> None:
        """Toggle isolation-driven contact reduction for this city's engine."""
        self.isolated = isolated
        self.engine.set_isolated(isolated)

    #
    # Travel support (relocating residents; hosting visitors)
    #
    def checkout(self, individual_id: int) -> DiseaseToken:
        """Mark a resident as away and return a token carrying their state.

        While away the individual is skipped by this city's transmission and
        progression (they are not physically here) but remains in the census so
        their evolving state can still be mirrored via :meth:`sync_absent`.
        """
        ind = self.engine.individuals[individual_id]
        ind.present = False
        return DiseaseToken(state=ind.state, days_in_state=ind.days_in_state,
                            infected_by=ind.infected_by,
                            infection_generation=ind.infection_generation,
                            infection_day=ind.infection_day,
                            recovery_day=ind.recovery_day)

    def sync_absent(self, individual_id: int, token: DiseaseToken) -> None:
        """Mirror an away resident's evolving state into the census (stays away)."""
        ind = self.engine.individuals[individual_id]
        ind.state = token.state
        ind.days_in_state = token.days_in_state
        ind.infected_by = token.infected_by
        ind.infection_generation = token.infection_generation
        ind.infection_day = token.infection_day
        ind.recovery_day = token.recovery_day

    def checkin(self, individual_id: int, token: DiseaseToken) -> None:
        """Return an away resident home, writing back their final state."""
        ind = self.engine.individuals[individual_id]
        ind.state = token.state
        ind.days_in_state = token.days_in_state
        ind.infected_by = token.infected_by
        ind.infection_generation = token.infection_generation
        ind.infection_day = token.infection_day
        ind.recovery_day = token.recovery_day
        ind.present = True

    def host_visitor_day(self, token: DiseaseToken, rng: Generator, day: int,
                         visitor_global_id: str) -> VisitDayResult:
        """Run one day for a visitor in this city, then age their disease.

        Interaction is network-based (the visitor shares a random host's
        contacts) and bidirectional, mirroring resident transmission: an
        infectious visitor may expose susceptible residents, and a susceptible
        visitor may be infected by infectious residents. The visitor's own
        state then progresses one day. All disease timing and probabilities
        come from this city's engine -- the travel layer supplies only the
        traveler and the RNG. Behavioral-response/isolation contact reduction
        (see :meth:`engine.DiseaseEngine.subsample_contacts`) applies to an
        infectious visitor exactly as it would to a resident of this city.

        Args:
            token: The visitor's disease token (mutated in place).
            rng: The generator to draw interaction outcomes from.
            day: The current simulated day (for transmission-tracking stamps).
            visitor_global_id: The visitor's global id (``"{home_city}-{id}"``),
                stamped as ``infected_by`` on any resident they expose.

        Returns:
            A :class:`VisitDayResult` describing this day's transmissions.
        """
        p = self.config.infection_probability
        host_id = int(rng.integers(0, self.config.population_size))
        contacts = np.unique(np.append(self.contacts_of(host_id, rng), host_id))

        infected: List[int] = []
        acquired_from: Optional[int] = None

        if token.state is State.INFECTIOUS:
            contacts = self.engine.subsample_contacts(contacts, rng)
            for cid in contacts:
                resident = self.engine.individuals[int(cid)]
                if (resident.present and resident.state is State.SUSCEPTIBLE
                        and rng.random() < p
                        and self.expose(int(cid), infected_by=visitor_global_id,
                                        infection_generation=token.infection_generation + 1,
                                        infection_day=day)):
                    infected.append(int(cid))
        elif token.state is State.SUSCEPTIBLE:
            for cid in contacts:
                resident = self.engine.individuals[int(cid)]
                if (resident.present and resident.state is State.INFECTIOUS
                        and rng.random() < p):
                    token.state = State.EXPOSED
                    token.days_in_state = 0
                    token.infected_by = f"{self.id}-{cid}"
                    token.infection_generation = resident.infection_generation + 1
                    token.infection_day = day
                    acquired_from = int(cid)
                    break

        # Age the visitor's disease one day (E -> I -> R), same timing as home
        prev_state = token.state
        token.state, token.days_in_state = self.engine.advance_state(
            token.state, token.days_in_state)
        if prev_state is State.INFECTIOUS and token.state is State.RECOVERED:
            token.recovery_day = day
        return VisitDayResult(infected_resident_ids=infected,
                              acquired_from=acquired_from)

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
                "peak_recovered": 0.0,
                "peak_recovered_day": -1.0,
                "day_outbreak_began": -1.0,
                "day_outbreak_peaked": -1.0,
                "imported_infections": float(self.imported_infections),
            }

        final = self.history[-1]
        population = self.config.population_size

        # Total infected = everyone who left susceptible
        total_infected = population - final.susceptible
        attack_rate = total_infected / population if population else 0.0

        peak_inf = max(self.history, key=lambda r: r.infectious)
        peak_exp = max(self.history, key=lambda r: r.exposed)
        peak_rec = max(self.history, key=lambda r: r.recovered)

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
            "peak_recovered": float(peak_rec.recovered),
            "peak_recovered_day": float(peak_rec.day),
            # Convenience aliases requested by the milestone spec.
            "day_outbreak_began": float(first_infection_day),
            "day_outbreak_peaked": float(peak_inf.day),
            "imported_infections": float(self.imported_infections),
        }
