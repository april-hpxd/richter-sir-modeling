"""Pre-outbreak vaccination: an optional, separate intervention.

Milestone 1 scope only: vaccine effectiveness is 100%, vaccination happens
once before the simulation's first day (before any seeding or stepping), and
a vaccinated individual stays vaccinated (``State.VACCINATED``) for the whole
run -- it can never become infected (the engine's transmission step only
targets ``State.SUSCEPTIBLE`` contacts) and, since it never becomes
``INFECTIOUS``, it can never transmit either. See ``disease_model.State`` for
the state-machine note and ``README.md`` for the user-facing explanation.

Deliberately does not implement: waning immunity, boosters, partial efficacy,
hesitancy, demographic targeting, mid-outbreak vaccination, or multiple
vaccine types -- those are later milestones.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from numpy.random import Generator

from disease_model import Individual, State

VACCINATION_STRATEGIES = ("random",)


def vaccinate_random(individuals: Sequence[Individual], count: int,
                     rng: Generator) -> List[int]:
    """Vaccinate ``count`` individuals chosen uniformly at random.

    Only individuals currently ``SUSCEPTIBLE`` are eligible, so this is safe
    to call before any disease seeding (nobody has been infected yet, so
    "eligible" and "susceptible" coincide) and will never vaccinate someone
    already infected. Each eligible individual is selected at most once.

    Args:
        individuals: The population to vaccinate (e.g. an
            :class:`~engine.DiseaseEngine`'s ``individuals`` list). Mutated
            in place: chosen individuals are set to ``State.VACCINATED``.
        count: Exactly how many individuals to vaccinate.
        rng: The shared random generator (reuses the simulation's/city's own
            generator, so a run's ``random_seed`` alone makes the vaccinated
            set reproducible, exactly like seeding and contact generation).

    Returns:
        The sorted list of individual ids that were vaccinated.

    Raises:
        ValueError: If ``count`` is negative, or exceeds the number of
            currently susceptible (eligible) individuals.
    """
    if count < 0:
        raise ValueError("count must be >= 0.")
    eligible = [ind.id for ind in individuals if ind.state is State.SUSCEPTIBLE]
    if count > len(eligible):
        raise ValueError(
            f"Cannot vaccinate {count}; only {len(eligible)} eligible "
            "(susceptible) individuals available."
        )
    if count == 0:
        return []
    chosen = rng.choice(eligible, size=count, replace=False)
    chosen_ids = sorted(int(c) for c in chosen)
    for individual_id in chosen_ids:
        ind = individuals[individual_id]
        ind.state = State.VACCINATED
        ind.days_in_state = 0
    return chosen_ids


def vaccination_report(enabled: bool, strategy: str,
                       vaccinated_ids: Sequence[int],
                       population_size: int) -> Dict[str, object]:
    """Summarise a (possibly empty) vaccination for statistics/reporting.

    Args:
        enabled: Whether vaccination was configured for this run.
        strategy: The vaccination strategy used (e.g. ``"random"``).
        vaccinated_ids: The ids returned by the vaccination method.
        population_size: Total population, for the coverage fraction.

    Returns:
        Dict with ``vaccination_enabled``, ``vaccination_strategy``,
        ``number_vaccinated``, and ``vaccination_coverage``.
    """
    number_vaccinated = len(vaccinated_ids)
    coverage = number_vaccinated / population_size if population_size else 0.0
    return {
        "vaccination_enabled": enabled,
        "vaccination_strategy": strategy if enabled else None,
        "number_vaccinated": number_vaccinated,
        "vaccination_coverage": coverage,
    }
