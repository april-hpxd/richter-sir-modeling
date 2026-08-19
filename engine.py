"""The SEIR disease engine: progression and transmission, one day at a time.

Worked example (incubation 2, infectious 6), for a person exposed on day 0::

    day 0:  S -> E      (seeded / newly exposed)
    day 2:  E -> I      (after 2 incubation days)
    day 8:  I -> R      (after 6 infectious days)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.random import Generator

from disease_model import Individual, State
from interaction import ContactModel


class DiseaseEngine:
    """Advance an SEIR epidemic day-by-day over a well-mixed or networked city.

    Attributes:
        individuals: The population, indexed by id (``individuals[i].id == i``).
        contact_model: Supplies each infectious individual's daily contacts.
        infection_probability: Per-interaction transmission probability.
        incubation_days: Days in ``EXPOSED`` before becoming ``INFECTIOUS``.
        infectious_days: Days in ``INFECTIOUS`` before ``RECOVERED``.
        day: Number of completed simulated days (``0`` before any step).
    """

    def __init__(
        self,
        population_size: int,
        contact_model: ContactModel,
        infection_probability: float,
        incubation_days: int,
        infectious_days: int,
        rng: Generator,
        id_prefix: str = "",
        behavioral_response_factor: Optional[float] = None,
        isolation_contact_multiplier: float = 1.0,
    ) -> None:
        """Initialise the engine with everyone Susceptible.

        Args:
            population_size: Number of individuals to create.
            contact_model: The interaction layer (well-mixed here; a contact
                network in a later milestone).
            infection_probability: Per-interaction transmission probability.
            incubation_days: Duration of the ``EXPOSED`` period.
            infectious_days: Duration of the ``INFECTIOUS`` period.
            rng: The single shared NumPy generator driving all randomness.
            id_prefix: Prepended to local ids when stamping ``infected_by``,
                e.g. ``"city0-"`` in a regional run, so ids stay globally
                unique across cities. Empty for a bare single-city run.
            behavioral_response_factor: If set, an infectious individual's
                daily contacts are subsampled to this fraction of normal
                (e.g. ``0.5`` halves them). ``None`` disables the behavior.
            isolation_contact_multiplier: Extra multiplier applied on top of
                ``behavioral_response_factor`` while this city is isolated
                (see :meth:`set_isolated`); ``1.0`` is a no-op.
        """
        self.individuals: List[Individual] = [
            Individual(id=i) for i in range(population_size)
        ]
        self.contact_model = contact_model
        self.infection_probability = infection_probability
        self.incubation_days = incubation_days
        self.infectious_days = infectious_days
        self._rng = rng
        self.day: int = 0
        self.id_prefix = id_prefix
        self.behavioral_response_factor = behavioral_response_factor
        self.isolation_contact_multiplier = isolation_contact_multiplier
        self.isolated = False
        # Actual contact-array length drawn by each infectious individual on
        # the most recent step(); read by the node export for `contacts_today`.
        self.last_contact_counts: Dict[int, int] = {}

    # 
    # Seeding
    # 
    def seed_exposed(self, count: int) -> List[int]:
        """Seed ``count`` initial cases in the ``EXPOSED`` state.

        Initial cases start as ``EXPOSED`` (not ``INFECTIOUS``) so that the
        outbreak begins with the natural ``S -> E`` step of a patient-zero
        timeline and the seeds incubate before they can transmit.

        Args:
            count: Number of individuals to expose.

        Returns:
            The list of individual ids that were exposed.

        Raises:
            ValueError: If ``count`` exceeds the number of susceptibles.
        """
        susceptible = [ind.id for ind in self.individuals
                       if ind.state is State.SUSCEPTIBLE]
        if count > len(susceptible):
            raise ValueError(
                f"Cannot seed {count} cases; only {len(susceptible)} "
                "susceptible individuals available."
            )
        chosen = self._rng.choice(susceptible, size=count, replace=False)
        for cid in chosen:
            ind = self.individuals[int(cid)]
            ind.state = State.EXPOSED
            ind.days_in_state = 0
            ind.infected_by = None
            ind.infection_generation = 0
            ind.infection_day = self.day
        return [int(c) for c in chosen]

    # 
    # Daily update
    # 
    def step(self) -> Dict[str, int]:
        """Advance the epidemic by exactly one day.

        Applies transmission then progression as described in the module
        docstring, both decided from the start-of-day state.

        Returns:
            A dict with the day's flows: ``new_exposed``, ``new_infectious``,
            ``new_recovered``, and ``transmissions`` (a list of
            ``(source_id, target_id)`` pairs newly exposed today, for
            visualization and transmission tracking).
        """
        newly_exposed = self._transmit()
        new_infectious, new_recovered = self._progress()

        # Commit new exposures last, so they are not aged this day.
        next_day = self.day + 1
        for target_id, source_id in newly_exposed:
            ind = self.individuals[target_id]
            ind.state = State.EXPOSED
            ind.days_in_state = 0
            ind.infection_day = next_day
            if source_id is None:
                ind.infected_by = None
                ind.infection_generation = 0
            else:
                source = self.individuals[source_id]
                ind.infected_by = f"{self.id_prefix}{source_id}"
                ind.infection_generation = source.infection_generation + 1

        self.day = next_day
        return {
            "new_exposed": len(newly_exposed),
            "new_infectious": new_infectious,
            "new_recovered": new_recovered,
            "transmissions": [(s, t) for t, s in newly_exposed],
        }

    def _transmit(self) -> List[Tuple[int, Optional[int]]]:
        """Compute today's new exposures from start-of-day infectious contacts.

        Every individual infectious at the start of the day meets the contacts
        supplied by the interaction layer (via :meth:`effective_contacts`, so
        any configured behavioral response is applied); each susceptible
        contact is exposed with probability ``infection_probability``.
        Recovered and already-exposed contacts are immune to (re)infection.

        Returns:
            A list of ``(target_id, source_id)`` pairs newly exposed today
            (each target listed once, even if contacted by several infectious
            individuals -- attributed to whichever source triggered it first).
        """
        infectious_ids = [ind.id for ind in self.individuals
                          if ind.state is State.INFECTIOUS and ind.present]
        newly_exposed: List[Tuple[int, Optional[int]]] = []
        exposed_set = set()
        contact_counts: Dict[int, int] = {}

        for src in infectious_ids:
            contacts = self.effective_contacts(src, self._rng)
            contact_counts[src] = len(contacts)
            for target_id in contacts:
                target = self.individuals[int(target_id)]
                if (target.state is State.SUSCEPTIBLE and target.present
                        and target.id not in exposed_set):
                    if self._rng.random() < self.infection_probability:
                        exposed_set.add(target.id)
                        newly_exposed.append((target.id, src))
        self.last_contact_counts = contact_counts
        return newly_exposed

    def nominal_contacts(self, individual_id: int) -> int:
        """Return this individual's normal (unreduced) daily contact count.

        For network-based models this is their persistent graph degree; for
        well-mixed it is the configured ``daily_contacts``. Used by the node
        export as `contacts_today` for anyone who wasn't infectious today
        (and therefore has no drawn count in :attr:`last_contact_counts`).
        """
        graph = getattr(self.contact_model, "graph", None)
        if graph is not None:
            return int(graph.degree(individual_id))
        return int(getattr(self.contact_model, "daily_contacts", 0))

    def effective_contacts(self, individual_id: int, rng: Generator) -> np.ndarray:
        """Return the contacts an infectious individual uses today.

        Wraps :attr:`contact_model` with the optional behavioral-response and
        isolation subsampling, so every call site (in-city transmission and
        hosted travelers alike) reduces contacts identically once someone is
        infectious. With no behavioral response configured and the city not
        isolated, this is exactly ``contact_model.contacts(...)``.

        Args:
            individual_id: The (infectious) individual seeking contacts.
            rng: The shared random generator.

        Returns:
            A 1-D array of other individual ids.
        """
        contacts = self.contact_model.contacts(individual_id, rng)
        return self.subsample_contacts(contacts, rng)

    def subsample_contacts(self, contacts: np.ndarray, rng: Generator) -> np.ndarray:
        """Apply this engine's behavioral-response/isolation reduction to a
        pre-computed contact array.

        Used by :meth:`effective_contacts` for residents' in-city contacts.
        Travel (:meth:`city.City.host_visitor_day`) does not use this: an
        inter-city visit is already a single fixed contact, independent of
        this city's network and its behavioral-response/isolation settings.

        Args:
            contacts: The full, un-reduced contact array.
            rng: The shared random generator.

        Returns:
            ``contacts``, or a subsampled subset if a reduction is active.
        """
        factor = 1.0
        if self.behavioral_response_factor is not None:
            factor *= self.behavioral_response_factor
        if self.isolated:
            factor *= self.isolation_contact_multiplier
        if factor >= 1.0 or len(contacts) == 0:
            return contacts
        k = max(1, round(len(contacts) * factor))
        if k >= len(contacts):
            return contacts
        chosen = rng.choice(len(contacts), size=k, replace=False)
        return contacts[chosen]

    def set_isolated(self, isolated: bool) -> None:
        """Toggle whether this engine's population is under isolation.

        While isolated, :meth:`effective_contacts` applies
        ``isolation_contact_multiplier`` in addition to any behavioral
        response. Travel restrictions during isolation are handled
        separately by :class:`~travel.TravelManager`.
        """
        self.isolated = isolated

    def _progress(self) -> tuple[int, int]:
        """Age start-of-day infected individuals and apply ``E->I``/``I->R``.

        Returns:
            A ``(new_infectious, new_recovered)`` count pair for the day.
        """
        new_infectious = 0
        new_recovered = 0
        for ind in self.individuals:
            if not ind.present:
                continue
            if ind.state is State.EXPOSED:
                ind.days_in_state += 1
                if ind.days_in_state >= self.incubation_days:
                    ind.state = State.INFECTIOUS
                    ind.days_in_state = 0
                    new_infectious += 1
            elif ind.state is State.INFECTIOUS:
                ind.days_in_state += 1
                if ind.days_in_state >= self.infectious_days:
                    ind.state = State.RECOVERED
                    ind.days_in_state = 0
                    ind.recovery_day = self.day + 1
                    new_recovered += 1
        return new_infectious, new_recovered

    # 
    # Convenience accessors
    # 
    def counts(self) -> Dict[str, int]:
        """Return the current number of individuals in each compartment.

        Returns:
            Dict with integer keys ``"S"``, ``"E"``, ``"I"``, ``"R"``.
        """
        tally = {s: 0 for s in State}
        for ind in self.individuals:
            tally[ind.state] += 1
        return {
            "S": tally[State.SUSCEPTIBLE],
            "E": tally[State.EXPOSED],
            "I": tally[State.INFECTIOUS],
            "R": tally[State.RECOVERED],
        }

    def states(self) -> List[State]:
        """Return a snapshot list of every individual's state, ordered by id.

        Used by the visualization to colour each person; returned as a fresh
        list so callers can store per-day frames without aliasing live state.
        """
        return [ind.state for ind in self.individuals]

    def is_epidemic_active(self) -> bool:
        """Return ``True`` while any individual is ``EXPOSED`` or ``INFECTIOUS``.

        Once ``False`` the state can no longer change, so the driver may stop.
        """
        return any(ind.state in (State.EXPOSED, State.INFECTIOUS)
                   for ind in self.individuals)

    def advance_state(self, state: State, days_in_state: int) -> tuple[State, int]:
        """Apply one day of ``E -> I -> R`` progression to a detached state.

        This is the same time-based progression used for residents, exposed as
        a pure function so the travel layer can age a person's disease while
        they are away from home *without* re-implementing disease timing. It
        performs no transmission.

        Args:
            state: The current disease state.
            days_in_state: Days already spent in that state.

        Returns:
            The ``(state, days_in_state)`` after advancing one day.
        """
        if state is State.EXPOSED:
            days_in_state += 1
            if days_in_state >= self.incubation_days:
                return State.INFECTIOUS, 0
            return State.EXPOSED, days_in_state
        if state is State.INFECTIOUS:
            days_in_state += 1
            if days_in_state >= self.infectious_days:
                return State.RECOVERED, 0
            return State.INFECTIOUS, days_in_state
        return state, days_in_state
