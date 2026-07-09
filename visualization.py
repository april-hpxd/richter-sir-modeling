"""Visualization: live animations and post-run summary plots.

This module is purely presentational. It reads a completed (or ready-to-run)
:class:`~simulation.Simulation` and its recorded history and renders it; it
never advances the disease itself. That separation keeps the scientific core
(:mod:`sir_engine`) independent of matplotlib.


* func:`animate_network` -- the primary view: the small-world contact graph
  with nodes coloured by compartment, animated day by day, with a live
  statistics panel. The layout is fixed, so the graph stays spatially
  consistent and the eye can follow infection spreading along edges.
* func:`animate_geographic` -- a second, "city-like" view. Each person becomes
  a Voronoi region tiling an approximate city boundary; as the epidemic
  spreads the map reddens and then greens, resembling an outbreak moving across
  a city. This is the harder, optional visualization; it is built on the same
  history so it can be refined later without touching the model.
* func:`plot_summary` -- static end-of-run figures (SIR curves, daily new
  infections, network infection percentage, and a peak/duration summary).

Colour convention (used everywhere): Susceptible = light grey, Infected = red,
Recovered = green.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Patch, Polygon as MplPolygon
from scipy.spatial import Voronoi

from config import Config
from simulation import DayRecord, Simulation
from sir_engine import State
from statistics import format_live_panel

# --- Shared colour scheme -----------------------------------------------------
COLOR_SUSCEPTIBLE = "#d9d9d9"  # light grey
COLOR_INFECTED = "#e63946"     # red
COLOR_RECOVERED = "#2a9d8f"    # green
_STATE_COLOR: Dict[State, str] = {
    State.SUSCEPTIBLE: COLOR_SUSCEPTIBLE,
    State.INFECTED: COLOR_INFECTED,
    State.RECOVERED: COLOR_RECOVERED,
}

Point = Tuple[float, float]
