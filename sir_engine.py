"""The core network SIR disease engine.

This is the heart of Milestone 1 -- and the piece explicitly intended to be
reused for every city in the future regional model. It knows how to advance a
Susceptible-Infected-Recovered epidemic by one day on an arbitrary contact
network, and nothing else. It has **no** dependency on visualization,
plotting, file I/O, or even on how the network was built. That isolation is
what makes it a clean, reusable "disease engine".

Model
-----
Every individual is in exactly one compartment:

* ``S`` -- Susceptible: can catch the disease.
* ``I`` -- Infected: currently infectious, can transmit to neighbours.
* ``R`` -- Recovered: immune; cannot be reinfected.

One simulated day (:meth:`SIREngine.step`) applies a **synchronous** update:

1. *Transmission.* Every edge connecting an Infected node to a Susceptible
   node is an independent transmission trial that succeeds with probability
   ``infection_probability``. A Susceptible with ``j`` infected neighbours is
   therefore infected today with probability ``1 - (1 - p) ** j``.
2. *Recovery.* Every node that was Infected at the *start* of the day recovers
   with probability ``recovery_probability``.

Both changes are computed from the start-of-day state and applied together at
the end of the day. Consequently a node infected *today* neither transmits nor
recovers until the *next* day -- the standard, well-defined discrete-time SIR
update that avoids order-dependent artefacts.
"""

from __future__ import annotations

import enum


class State(enum.Enum):
    """The three SIR compartments.

    Backed by short string values so they serialize cleanly (e.g. to CSV) and
    read well in logs, while still being a proper enum for safe comparisons.
    """

    SUSCEPTIBLE = "S"
    INFECTED = "I"
    RECOVERED = "R"


class SIREngine:
    """Advance an SIR epidemic day-by-day over a fixed contact network.

    The engine owns the mutable disease state of every node. The network's
    *structure* (edges, positions) is treated as read-only, so the same graph
    could in principle be shared, though each city normally owns its own.

    Attributes:
        graph: The contact network the epidemic runs on.
        infection_probability: Per-contact, per-day transmission probability.
        recovery_probability: Per-day recovery probability.
        state: Mapping from node id to its current :class:`State`.
        day: Number of completed simulated days (0 before any step).
    """
