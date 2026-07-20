"""Configuration for the single-city SEIR simulation.

All tunable parameters live in one immutable :class:`Config` object so that a
run is fully described by a single value and identical configurations (same
``random_seed`` included) reproduce identical simulations.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict


@dataclass(frozen=True)
class Config:
    """Immutable bundle of simulation parameters.

    Attributes:
        population_size: Number of individuals per city (for single-city runs).
        daily_contacts: How many other people each *infectious* individual
            interacts with per day in the contact model.
        infection_probability: Probability that a single S<-I interaction
            results in transmission (the new case enters ``EXPOSED``).
        incubation_days: Days spent ``EXPOSED`` (infected but not yet
            infectious) before becoming ``INFECTIOUS``.
        infectious_days: Days spent ``INFECTIOUS`` before ``RECOVERED``.
        simulation_days: Upper bound on days to simulate; the run also stops
            early once no ``EXPOSED`` or ``INFECTIOUS`` individuals remain.
        random_seed: Seed for the single NumPy ``Generator`` driving *all*
            randomness, guaranteeing reproducibility.
        initial_infected: Number of initial cases (seeded as ``EXPOSED`` on
            day 0, matching the ``S -> E`` start of a patient-zero timeline).
        contact_model: Which contact model to use: "random-network",
            "well-mixed", or "watts-strogatz".
        random_degree_min/random_degree_max: Inclusive bounds for the number
            of persistent contacts assigned to every person in a seeded random
            contact graph.
        watts_strogatz_k: For Watts-Strogatz networks, the neighbourhood size.
        watts_strogatz_p: For Watts-Strogatz networks, the rewiring probability.

    Regional parameters (used by :class:`RegionalSimulation`):
        number_of_cities: How many independent cities in the regional simulation.
        population_per_city: Number of individuals per city (overrides
            ``population_size`` when running regionally).
        travel_fraction: Fraction of the population eligible to travel (0.0 to
            0.5, so at most half a city's residents can travel).
        daily_travel_rate: Fraction of eligible travelers who actually travel
            on a given day (0.0 to 1.0).

    A rough basic reproduction number is::

        R0 ~= infection_probability * mean_contacts * infectious_days

    where ``mean_contacts`` is the mean degree for a network model (or
    ``daily_contacts`` when well-mixed). With the defaults this is about 1.4 --
    enough to produce a clear outbreak in the
    seeded city and to reliably carry infection to the other city via travel,
    while still leaving some individuals uninfected.
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

    # --- Regional simulation parameters ----------------------------------
    number_of_cities: int = 2
    population_per_city: int = 50
    travel_fraction: float = 0.5
    daily_travel_rate: float = 0.1

    def __post_init__(self) -> None:
        """Validate parameters, raising ``ValueError`` on nonsensical input.

        Failing at construction time keeps a broken configuration from silently
        producing meaningless results.
        """
        if self.population_size < 2:
            raise ValueError("population_size must be >= 2.")
        if not 1 <= self.daily_contacts <= self.population_size - 1:
            raise ValueError(
                "daily_contacts must be between 1 and population_size - 1."
            )
        if self.contact_model not in ("random-network", "well-mixed", "watts-strogatz"):
            raise ValueError(
                "contact_model must be 'random-network', 'well-mixed', "
                "or 'watts-strogatz'."
            )
        if self.random_degree_min < 1:
            raise ValueError("random_degree_min must be >= 1.")
        if self.random_degree_max < self.random_degree_min:
            raise ValueError(
                "random_degree_max must be >= random_degree_min."
            )
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
        if not 1 <= self.initial_infected <= self.population_size:
            raise ValueError(
                "initial_infected must be between 1 and population_size."
            )
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

    def with_overrides(self, **overrides: Any) -> "Config":
        """Return a new :class:`Config` with the given fields replaced.

        Because :class:`Config` is frozen, this is the supported way to derive
        a variant (e.g. a more transmissible scenario)::

            fast = base.with_overrides(infection_probability=0.1)

        Args:
            **overrides: Field names mapped to new values.

        Returns:
            A validated new configuration instance.
        """
        return replace(self, **overrides)

    def as_dict(self) -> Dict[str, Any]:
        """Return the configuration as a plain dictionary (for logging/export)."""
        return asdict(self)

    def estimated_r0(self) -> float:
        """Return a back-of-the-envelope basic reproduction number.

        Not used to drive the simulation; provided only to help interpret
        parameters. The effective daily contact count depends on the active
        contact model: for Watts-Strogatz it is the mean network degree
        (``k``), whereas the well-mixed model uses ``daily_contacts``::

            R0 ~= infection_probability * mean_contacts * infectious_days
        """
        if self.contact_model == "watts-strogatz":
            mean_contacts = self.watts_strogatz_k
        elif self.contact_model == "random-network":
            mean_contacts = (self.random_degree_min + self.random_degree_max) / 2
        else:
            mean_contacts = self.daily_contacts
        return (self.infection_probability * mean_contacts
                * self.infectious_days)
