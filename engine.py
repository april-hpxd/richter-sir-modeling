"""The SEIR disease engine: progression and transmission, one day at a time.

1. **Transmission** -- infectious individuals meet contacts (supplied by a
   :class:`~interaction.ContactModel`) and may expose susceptible ones.
2. **Progression** -- each infected individual moves ``E -> I -> R`` purely as
   a function of how many days it has spent in its current state.

Worked example (incubation 2, infectious 6), for a person exposed on day 0::

    day 0:  S -> E      (seeded / newly exposed)
    day 2:  E -> I      (after 2 incubation days)
    day 8:  I -> R      (after 6 infectious days)
"""

from __future__ import annotations

from typing import Dict, List

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

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
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
            self.individuals[int(cid)].state = State.EXPOSED
            self.individuals[int(cid)].days_in_state = 0
        return [int(c) for c in chosen]

    # ------------------------------------------------------------------
    # Daily update
    # ------------------------------------------------------------------
    def step(self) -> Dict[str, int]:
        """Advance the epidemic by exactly one day.

        Applies transmission then progression as described in the module
        docstring, both decided from the start-of-day state.

        Returns:
            A dict with the day's flows: ``new_exposed``, ``new_infectious``,
            and ``new_recovered``.
        """
        newly_exposed = self._transmit()
        new_infectious, new_recovered = self._progress()

        # Commit new exposures last, so they are not aged this day.
        for cid in newly_exposed:
            self.individuals[cid].state = State.EXPOSED
            self.individuals[cid].days_in_state = 0

        self.day += 1
        return {
            "new_exposed": len(newly_exposed),
            "new_infectious": new_infectious,
            "new_recovered": new_recovered,
        }

    def _transmit(self) -> List[int]:
        """Compute today's new exposures from start-of-day infectious contacts.

        Every individual infectious at the start of the day meets the contacts
        supplied by the interaction layer; each susceptible contact is exposed
        with probability ``infection_probability``. Recovered and already-
        exposed contacts are immune to (re)infection.

        Returns:
            The ids of individuals newly exposed today (each listed once, even
            if contacted by several infectious individuals).
        """
        infectious_ids = [ind.id for ind in self.individuals
                          if ind.state is State.INFECTIOUS]
        newly_exposed: List[int] = []
        exposed_set = set()

        for src in infectious_ids:
            for target_id in self.contact_model.contacts(src, self._rng):
                target = self.individuals[int(target_id)]
                if (target.state is State.SUSCEPTIBLE
                        and target.id not in exposed_set):
                    if self._rng.random() < self.infection_probability:
                        exposed_set.add(target.id)
                        newly_exposed.append(target.id)
        return newly_exposed

    def _progress(self) -> tuple[int, int]:
        """Age start-of-day infected individuals and apply ``E->I``/``I->R``.

        Returns:
            A ``(new_infectious, new_recovered)`` count pair for the day.
        """
        new_infectious = 0
        new_recovered = 0
        for ind in self.individuals:
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
                    new_recovered += 1
        return new_infectious, new_recovered

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
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
