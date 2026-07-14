"""Core disease-domain data types: the SEIR states and the individual.

This module holds *only* the vocabulary of the disease model -- the compartment
:class:`State` and the per-person :class:`Individual` record. It contains no
dynamics (that lives in :mod:`engine`), no randomness, and no plotting, so it
can be imported freely by the engine, statistics, and visualization without
creating circular dependencies or coupling those layers together.

SEIR compartments
-----------------
* ``S`` Susceptible  -- can catch the disease.
* ``E`` Exposed      -- infected and incubating, but **not yet infectious**.
* ``I`` Infectious   -- can transmit to susceptible contacts.
* ``R`` Recovered    -- permanently immune; cannot infect or be infected.

Adding the ``E`` (Exposed) compartment is the key change from a plain SIR
model: a newly infected person spends an incubation period unable to infect
others before becoming infectious.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


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
    """

    id: int
    state: State = State.SUSCEPTIBLE
    days_in_state: int = 0
