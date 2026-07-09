"""Configuration for the single-city network SIR simulation.

Stores every simulation parameter in one immutable configuration object.
Design note for future multi-city work
---
In Milestone 1 there is exactly one :class:`Config` per run. In the eventual
regional model, each ``City`` object will own its *own* :class:`Config`
instance (different populations, contact rates, etc.), and a separate mobility
layer will sit on top. Nothing in this class assumes a single global city, so
it can be instantiated many times without modification.


Fields:
population_size: int
average_contacts: int
rewiring_probability: float
infection_probability: float
recovery_probability: float
initial_infected: int
simulation_days: int
random_seed: int
(animation_speed: int) if turning into gif after modeling

Possibly also:
show_geographic: bool
show_plots: bool
"""

from dataclasses import dataclass, asdict, replace

@dataclass(frozen=True)
class Config:
    """Immutable bundle of simulation parameters.

    Every value has a sensible default chosen to produce a clear, textbook
    epidemic curve on a small-world network. The class is frozen (immutable)
    so that a configuration cannot be accidentally mutated mid-run; use
    :meth:`with_overrides` to derive a modified copy.

    Epidemiological parameters
    ---
    infection_probability:
        Probability that a single Infected node transmits to a single
        Susceptible neighbour on a given day (a per-contact, per-day rate).
    recovery_probability:
        Probability that an Infected node recovers on a given day. Its
        reciprocal ``1 / recovery_probability`` is the mean infectious period
        in days.

    A useful sanity check is the basic reproduction number, approximated on a
    network as::

        R0 ~= infection_probability * average_contacts / recovery_probability

    With the defaults below this is roughly 3.0, which yields a pronounced
    outbreak that does not immediately infect everyone.
    """

    # --- Network / population --------------------------------------------
    population_size: int = 1000 # Number of nodes in one population

    average_contacts: int = 5 # Mean num of social contacts per person

    """
    Must be an even integer >= 2 and < ``population_size`` because the
    Watts-Strogatz construction connects each node to ``k/2`` neighbours on
    each side of a ring before rewiring.
    """

    rewiring_probability: float = 0.1
    """Watts-Strogatz rewiring probability (``beta``), in ``[0, 1]``.

    ``0`` gives a pure ring lattice (tight local communities, no shortcuts);
    ``1`` gives an essentially random graph. Small values (~0.1) create the
    "small-world" regime: strong local clustering (neighbourhoods, schools,
    workplaces) plus a few long-range links (occasional outside contact).
    """

    # --- Disease dynamics -------------------------------------------------
    infection_probability: float = 0.08 # Per-contact, per-day transmission probability in [0, 1]

    recovery_probability: float = 0.1 # Recovery confers immunity

    initial_infected: int = 5 # Start off a small number for sensible outcome

    # --- Run control ------------------------------------------------------
    simulation_days: int = 160 # Maximum days to model

    """
    The driver loop stops early once no infected individuals remain, so this is
    an upper bound; the default outbreak typically resolves well before it.
    """

    random_seed: int = 42 # Seed for all randomness (network build, seeding, transmission)

    """
    A fixed seed makes every run fully reproducible. Use ``None`` for
    nondeterministic behaviour.
    """
