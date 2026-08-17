"""Core disease-domain data types: the SEIR states and the individual.

SEIR compartments
* ``S`` Susceptible  -- can catch the disease.
* ``E`` Exposed      -- infected and incubating, but **not yet infectious**.
* ``I`` Infectious   -- can transmit to susceptible contacts.
* ``R`` Recovered    -- permanently immune; cannot infect or be infected.

"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional


class State(enum.Enum):
    """The four SEIR disease compartments.

    Backed by short string values (``"S"``, ``"E"``, ``"I"``, ``"R"``) so they
    read well in logs and serialize cleanly, while remaining a proper ``Enum``
    for safe, explicit comparisons (``ind.state is State.INFECTIOUS``).
    """

    SUSCEPTIBLE = "S"
    EXPOSED = "E"
    INFECTIOUS = "I"
    RECOVERED = "R"


@dataclass
class Individual:
    """One person in the simulated city.

    Each individual progresses through the disease independently, driven only
    by how long they have been in their current state. That is the entire
    per-person state the engine needs, and it is deliberately independent of
    *how* people come into contact -- so the same record works unchanged when
    the well-mixed interaction model is later replaced by a contact network.

    Attributes:
        id: Stable integer identifier, also the person's index in the
            population list and their fixed position in the visualization.
        state: The individual's current :class:`State`.
        days_in_state: Number of complete days spent in the current state.
            Reset to ``0`` on every transition; used to time ``E -> I`` and
            ``I -> R`` progressions.
        present: Whether the individual is currently participating in their
            home city's dynamics. Set ``False`` while the person is away
            travelling: they are still counted in the census (so their state
            can be mirrored and animated) but the engine neither transmits to
            nor progresses them locally -- the travel layer drives them while
            they are elsewhere. This is a generic presence concept; the engine
            never learns *why* someone is absent.
        infected_by: Global id of the source who exposed this individual, as
            ``"{city_id}-{local_id}"`` (or just the local id for a
            single-city run without a notion of city). ``None`` for a never-
            infected individual or an initial seeded case.
        infection_generation: Transmission generation: ``0`` for seeded
            cases, otherwise one more than the source's generation.
        infection_day: The day (matching :class:`~simulation.DailyRecord.day`)
            this individual became ``EXPOSED``, or ``None`` if never infected.
        recovery_day: The day this individual became ``RECOVERED``, or
            ``None`` if not yet recovered.
    """

    id: int
    state: State = State.SUSCEPTIBLE
    days_in_state: int = 0
    present: bool = True
    infected_by: Optional[str] = None
    infection_generation: int = 0
    infection_day: Optional[int] = None
    recovery_day: Optional[int] = None


@dataclass
class PersonSnapshot:
    """One individual's exportable state on one day.

    A lightweight, per-day copy of the fields the node-level CSV export
    (``node_export.py``) and the visualization need, kept separate from the
    live :class:`Individual` so recorded history doesn't alias mutable state.

    ``cluster_id`` is the person's cluster membership under the ``clustered``
    contact model (see ``interaction.ClusteredContactModel``), or ``None``
    for any other contact model.
    """

    id: int
    state: State
    days_in_state: int
    infected_by: Optional[str]
    infection_generation: int
    infection_day: Optional[int]
    recovery_day: Optional[int]
    contacts_today: int
    cluster_id: Optional[int] = None
