"""Configuration for the regional multi-city SEIR simulation.

All tunable parameters live in one immutable :class:`Config` object so that a
run is fully described by a single value and identical configurations (same
``random_seed`` included) reproduce identical simulations.

The configuration is fully *data-driven*: the number of cities, each city's
population, the (possibly asymmetric) travel matrix, the distribution of trip
lengths, and the visualization mode are all read from here. Changing only the
configuration -- a :class:`Config` literal, CLI flags, or a JSON file loaded
via :meth:`Config.from_json` -- rebuilds an entirely different regional
simulation with no source changes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

CONTACT_MODELS = ("random-network", "well-mixed", "watts-strogatz")
VISUALIZATION_MODES = ("auto", "network", "cluster", "heatmap", "pie")

# Thresholds used by "auto" mode to pick a visualization automatically from
# the largest city's population: <= NETWORK_MAX_POPULATION individual nodes,
# <= CLUSTER_MAX_POPULATION grouped into communities, otherwise population
# tiles per city (heatmap).
NETWORK_MAX_POPULATION = 150
CLUSTER_MAX_POPULATION = 1000


@dataclass(frozen=True)
class Config:
    """Immutable, fully data-driven bundle of simulation parameters.

    Population / interaction:
        population_size: Individuals in a single-city (``--single-city``) run.
        daily_contacts: Contacts per person per day (well-mixed model only).
        contact_model: One of :data:`CONTACT_MODELS`.
        random_degree_min/max: Inclusive degree bounds for ``random-network``.
        watts_strogatz_k/p: Mean degree and rewiring prob for ``watts-strogatz``.

    Disease dynamics:
        infection_probability: Per-interaction transmission probability.
        incubation_days: Days spent ``EXPOSED`` before ``INFECTIOUS``.
        infectious_days: Days spent ``INFECTIOUS`` before ``RECOVERED``.
        initial_infected: Cases seeded (as ``EXPOSED``) in city 0 on day 0.

    Run control:
        simulation_days: Upper bound on days; also stops when no E/I remain.
        random_seed: Master seed -- the single source of all randomness.

    Regional structure (all optional; sensible homogeneous defaults):
        number_of_cities: City count when ``city_populations`` is not given.
        population_per_city: Population when ``city_populations`` is not given.
        city_populations: Explicit per-city sizes, e.g. ``(500, 200, 1500)``.
            When set it defines both the number of cities and their sizes.
        travel_matrix: ``N x N`` daily per-eligible-traveler visit
            probabilities; ``travel_matrix[i][j]`` is the chance an eligible
            resident of city ``i`` visits city ``j`` today. The diagonal is
            ignored. Need not be symmetric. When ``None`` a uniform matrix is
            derived from ``daily_travel_rate``.
        travel_fraction: Fraction of each city eligible to travel (0..0.5).
        daily_travel_rate: Used only to build the default ``travel_matrix``.
        trip_duration_distribution: ``((days, weight), ...)`` describing how
            long trips last. Defaults to always-1-day (return next step).

    Output:
        visualization_mode: One of :data:`VISUALIZATION_MODES`.
        heatmap_tile_fraction: Approximate share of a city's population
            represented by one heatmap tile. The default ``0.05`` creates a
            stable 20-tile city block; each tile is coloured by its own
            infectious share.

    A rough basic reproduction number is
    ``R0 ~= infection_probability * mean_contacts * infectious_days`` where
    ``mean_contacts`` is the network's mean degree (or ``daily_contacts`` when
    well-mixed).
    """

    # --- Population / interaction ----------------------------------------
    population_size: int = 50
    daily_contacts: int = 8
    contact_model: str = "random-network"
    random_degree_min: int = 1
    random_degree_max: int = 7
    watts_strogatz_k: int = 8
    watts_strogatz_p: float = 0.1

    # --- Disease dynamics -------------------------------------------------
    infection_probability: float = 0.06
    incubation_days: int = 2
    infectious_days: int = 6
    initial_infected: int = 2

    # --- Run control ------------------------------------------------------
    simulation_days: int = 120
    random_seed: int = 42

    # --- Regional structure ----------------------------------------------
    number_of_cities: int = 2
    population_per_city: int = 50
    city_populations: Optional[Tuple[int, ...]] = None
    travel_matrix: Optional[Tuple[Tuple[float, ...], ...]] = None
    travel_fraction: float = 0.5
    daily_travel_rate: float = 0.1
    trip_duration_distribution: Tuple[Tuple[int, float], ...] = ((1, 1.0),)

    # --- Output -----------------------------------------------------------
    visualization_mode: str = "auto"
    heatmap_tile_fraction: float = 0.05

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Validate parameters, raising ``ValueError`` on nonsensical input."""
        # Normalise any list inputs (e.g. from JSON) to tuples so the frozen
        # instance stays hashable and immutable.
        object.__setattr__(self, "city_populations",
                           _as_int_tuple(self.city_populations))
        object.__setattr__(self, "travel_matrix",
                           _as_matrix(self.travel_matrix))
        object.__setattr__(self, "trip_duration_distribution",
                           _as_pairs(self.trip_duration_distribution))

        if self.population_size < 2:
            raise ValueError("population_size must be >= 2.")
        if not 1 <= self.daily_contacts <= self.population_size - 1:
            raise ValueError(
                "daily_contacts must be between 1 and population_size - 1.")
        if self.contact_model not in CONTACT_MODELS:
            raise ValueError(f"contact_model must be one of {CONTACT_MODELS}.")
        if self.random_degree_min < 1:
            raise ValueError("random_degree_min must be >= 1.")
        if self.random_degree_max < self.random_degree_min:
            raise ValueError("random_degree_max must be >= random_degree_min.")
        if self.watts_strogatz_k < 1:
            raise ValueError("watts_strogatz_k must be >= 1.")
        if not 0.0 <= self.watts_strogatz_p <= 1.0:
            raise ValueError("watts_strogatz_p must be in [0, 1].")
        if not 0.0 <= self.infection_probability <= 1.0:
            raise ValueError("infection_probability must be in [0, 1].")
        if self.incubation_days < 1:
            raise ValueError("incubation_days must be >= 1.")
        if self.infectious_days < 1:
            raise ValueError("infectious_days must be >= 1.")
        if self.simulation_days < 1:
            raise ValueError("simulation_days must be >= 1.")
        if self.number_of_cities < 1:
            raise ValueError("number_of_cities must be >= 1.")
        if self.population_per_city < 2:
            raise ValueError("population_per_city must be >= 2.")
        if not 0.0 <= self.travel_fraction <= 0.5:
            raise ValueError("travel_fraction must be in [0, 0.5].")
        if not 0.0 <= self.daily_travel_rate <= 1.0:
            raise ValueError("daily_travel_rate must be in [0, 1].")
        if self.visualization_mode not in VISUALIZATION_MODES:
            raise ValueError(
                f"visualization_mode must be one of {VISUALIZATION_MODES}.")
        if not 0.01 <= self.heatmap_tile_fraction <= 0.5:
            raise ValueError("heatmap_tile_fraction must be in [0.01, 0.5].")

        # Resolved regional structure.
        sizes = self.city_sizes()
        if any(p < 2 for p in sizes):
            raise ValueError("every city population must be >= 2.")
        if not 1 <= self.initial_infected <= sizes[0]:
            raise ValueError(
                "initial_infected must be between 1 and city 0's population.")

        if self.city_populations is not None:
            if len(self.city_populations) < 1:
                raise ValueError("city_populations must be non-empty.")

        if self.travel_matrix is not None:
            n = len(sizes)
            m = self.travel_matrix
            if len(m) != n or any(len(row) != n for row in m):
                raise ValueError(
                    "travel_matrix must be square with one row/column per city.")
            for row in m:
                for prob in row:
                    if not 0.0 <= prob <= 1.0:
                        raise ValueError("travel_matrix entries must be in [0, 1].")
            # Each city's total daily travel probability must not exceed 1.
            for i, row in enumerate(m):
                off_diagonal = sum(row) - row[i]
                if off_diagonal > 1.0 + 1e-9:
                    raise ValueError(
                        f"row {i} of travel_matrix sums to > 1 off-diagonal.")

        durations = self.trip_duration_distribution
        if not durations:
            raise ValueError("trip_duration_distribution must be non-empty.")
        for days, weight in durations:
            if days < 1:
                raise ValueError("trip durations must be >= 1 day.")
            if weight < 0:
                raise ValueError("trip duration weights must be >= 0.")
        if sum(w for _, w in durations) <= 0:
            raise ValueError("trip duration weights must sum to > 0.")

    # ------------------------------------------------------------------
    # Resolvers -- turn optional/loose fields into concrete structures
    # ------------------------------------------------------------------
    def city_sizes(self) -> Tuple[int, ...]:
        """Return the concrete per-city populations (index == city id)."""
        if self.city_populations is not None:
            return tuple(self.city_populations)
        return tuple(self.population_per_city for _ in range(self.number_of_cities))

    def num_cities(self) -> int:
        """Return the resolved number of cities."""
        return len(self.city_sizes())

    def travel_probability_matrix(self) -> np.ndarray:
        """Return the resolved ``N x N`` travel-probability matrix.

        When :attr:`travel_matrix` is unset, build a uniform matrix in which an
        eligible traveler visits *some* other city with total probability
        ``daily_travel_rate``, spread evenly across the other cities. The
        diagonal is always zero (you cannot "travel" to your own city).
        """
        n = self.num_cities()
        if self.travel_matrix is not None:
            matrix = np.array(self.travel_matrix, dtype=float)
        else:
            matrix = np.zeros((n, n), dtype=float)
            if n > 1:
                matrix[:] = self.daily_travel_rate / (n - 1)
        np.fill_diagonal(matrix, 0.0)
        return matrix

    def trip_durations(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(durations, probabilities)`` arrays for sampling trips."""
        days = np.array([d for d, _ in self.trip_duration_distribution],
                        dtype=np.int64)
        weights = np.array([w for _, w in self.trip_duration_distribution],
                           dtype=float)
        probs = weights / weights.sum()
        return days, probs

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def with_overrides(self, **overrides: Any) -> "Config":
        """Return a new :class:`Config` with the given fields replaced."""
        return replace(self, **overrides)

    def as_dict(self) -> Dict[str, Any]:
        """Return the configuration as a plain (JSON-serialisable) dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Build a :class:`Config` from a dict, ignoring unknown keys."""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load a :class:`Config` from a JSON file.

        This is the "change only the configuration file" entry point: the JSON
        may set any field, including ``city_populations`` and ``travel_matrix``
        as nested lists, e.g.::

            {
              "city_populations": [500, 200, 1500],
              "travel_matrix": [[0, 0.10, 0.02],
                                [0.08, 0, 0.05],
                                [0.01, 0.03, 0]],
              "trip_duration_distribution": [[1, 0.6], [3, 0.3], [7, 0.1]],
              "visualization_mode": "heatmap"
            }
        """
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def estimated_r0(self) -> float:
        """Return a back-of-the-envelope basic reproduction number."""
        if self.contact_model == "watts-strogatz":
            mean_contacts = self.watts_strogatz_k
        elif self.contact_model == "random-network":
            mean_contacts = (self.random_degree_min + self.random_degree_max) / 2
        else:
            mean_contacts = self.daily_contacts
        return self.infection_probability * mean_contacts * self.infectious_days


# ----------------------------------------------------------------------
# Small normalisation helpers (accept lists from JSON, store tuples)
# ----------------------------------------------------------------------
def _as_int_tuple(value: Optional[Sequence[int]]) -> Optional[Tuple[int, ...]]:
    if value is None:
        return None
    return tuple(int(v) for v in value)


def _as_matrix(value: Optional[Sequence[Sequence[float]]]
               ) -> Optional[Tuple[Tuple[float, ...], ...]]:
    if value is None:
        return None
    return tuple(tuple(float(x) for x in row) for row in value)


def _as_pairs(value: Sequence[Sequence[float]]) -> Tuple[Tuple[int, float], ...]:
    return tuple((int(d), float(w)) for d, w in value)
