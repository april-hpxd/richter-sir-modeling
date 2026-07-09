"""
Ties the three ingredients of Milestone 1 together:

* a contact network (from :mod:`network_generator`),
* an SIR disease engine (from :mod:`sir_engine`),
* and a recorded day-by-day history of the outbreak.

It exposes a small, deliberate API: build once, then either :meth:`run` the
whole epidemic or drive it one :meth:`step` at a time (the latter is what the
live animation uses). After each day it snapshots the compartment counts and a
copy of every node's state, so visualization and statistics can replay the
outbreak without re-running it.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DayRecord:
    """Immutable snapshot of the epidemic at the end of one day.

    Attributes:
        day: The day index (0 is the initial seeded state, before any step).
        susceptible: Number of Susceptible individuals.
        infected: Number of Infected individuals.
        recovered: Number of Recovered individuals.
        new_infections: Individuals who became infected *on* this day.
        new_recoveries: Individuals who recovered *on* this day.
        states: A copy of every node's :class:`State` at end of day, so the
            spatial spread can be re-drawn frame by frame after the run.
    """

    day: int
    susceptible: int
    infected: int
    recovered: int
    new_infections: int
    new_recoveries: int
    states: Dict[int, State] = field(repr=False)


class Simulation:
    """Run and record a single-city network SIR epidemic.

    Attributes:
        config: The configuration governing this run.
        graph: The contact network (built at construction).
        engine: The :class:`SIREngine` advancing the disease.
        history: List of :class:`DayRecord`, one per day including day 0.
    """
