"""Configuration for the single-city SEIR simulation.

All tunable parameters live in one immutable :class:`Config` object so that a
run is fully described by a single value and identical configurations (same
``random_seed`` included) reproduce identical simulations.

Scope of this milestone
------------------------
This milestone validates the **disease progression engine** on a small,
*well-mixed* virtual city (~50 people): every day anyone may interact with
anyone. The contact network (Watts-Strogatz) is intentionally **not** built
yet. Because who-meets-whom is isolated behind the interaction layer
(:mod:`interaction`), introducing the network later changes only that layer --
these parameters and the disease dynamics stay as they are.

Design note for future work
----------------------------
In the eventual regional model each ``City`` will own its own :class:`Config`
instance, and a mobility layer will sit on top. Nothing here assumes a single
global city, so :class:`Config` can be instantiated many times unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict


@dataclass(frozen=True)
class Config:
    """Immutable bundle of simulation parameters.

    Attributes:
        population_size: Number of individuals in the city.
        daily_contacts: How many other people each *infectious* individual
            interacts with per day in the well-mixed model.
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

    A rough basic reproduction number for a well-mixed population is::

        R0 ~= infection_probability * daily_contacts * infectious_days

    With the defaults this is about 2.4, enough to produce a clear outbreak in
    a fully susceptible population of 50 without instantly infecting everyone.
    """

    # --- Population / interaction ----------------------------------------
    population_size: int = 50
    daily_contacts: int = 8

    # --- Disease dynamics -------------------------------------------------
    infection_probability: float = 0.05
    incubation_days: int = 2
    infectious_days: int = 6
    initial_infected: int = 1

    # --- Run control ------------------------------------------------------
    simulation_days: int = 120
    random_seed: int = 42

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
        parameters::

            R0 ~= infection_probability * daily_contacts * infectious_days
        """
        return (self.infection_probability * self.daily_contacts
                * self.infectious_days)
