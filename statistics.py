"""Statistics and reporting for a completed (or in-progress) simulation.

This module turns a list of :class:`~simulation.DayRecord` snapshots into the
numbers the project cares about: peak infection, epidemic duration, attack
rate, daily new infections/recoveries, and a rough effective reproduction
number. It also handles CSV export.

Everything here is a pure function of the recorded history, so it can be called
mid-run (to feed the live statistics panel) or after the run (for the summary
and plots). It deliberately holds no state of its own.
"""

from __future__ import annotations
