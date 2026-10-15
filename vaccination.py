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

from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from numpy.random import Generator

from disease_model import Individual, State

VACCINATION_STRATEGIES = ("random", "high_degree", "high-degree")


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


def vaccinate_high_degree(individuals: Sequence[Individual], count: int,
                          rng: Generator, graph: nx.Graph) -> List[int]:
    """Vaccinate the ``count`` highest-degree eligible individuals.

    Degree is each individual's number of neighbors in the *pre-outbreak*
    contact network (``graph``), computed once here and never recalculated
    as the epidemic progresses -- this is a static, one-time targeting
    decision, not adaptive/contact-tracing-based targeting.

    Eligibility matches :func:`vaccinate_random`: only ``State.SUSCEPTIBLE``
    individuals, so this is only valid to call pre-outbreak (before seeding).

    Individuals are ranked by descending degree. When the cutoff degree value
    has more eligible individuals than the remaining budget, the tie is
    broken with a seeded random shuffle (using ``rng``) rather than relying
    on node/dict iteration order, so the result is reproducible given the
    same ``rng`` state but not an artifact of insertion order.

    Args:
        individuals: The population to vaccinate. Mutated in place.
        count: Exactly how many individuals to vaccinate.
        rng: The shared random generator (same one used for seeding/contacts),
            so results are reproducible given the same seed.
        graph: The pre-outbreak contact network (networkx Graph), one node
            per individual id.

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

    degree = dict(graph.degree())
    # Shuffle eligible ids first (seeded) so that any ties at the cutoff
    # degree are broken reproducibly and not by node/dict iteration order.
    shuffled = eligible.copy()
    rng.shuffle(shuffled)
    ranked = sorted(shuffled, key=lambda node_id: degree.get(node_id, 0),
                    reverse=True)
    chosen_ids = sorted(int(c) for c in ranked[:count])
    for individual_id in chosen_ids:
        ind = individuals[individual_id]
        ind.state = State.VACCINATED
        ind.days_in_state = 0
    return chosen_ids


def vaccinate(individuals: Sequence[Individual], count: int, rng: Generator,
             strategy: str = "random",
             graph: Optional[nx.Graph] = None) -> List[int]:
    """Dispatch to the configured vaccination strategy.

    This is the single entry point callers (e.g. ``City``) should use so
    strategy selection lives in one place instead of being duplicated at
    each call site.

    Args:
        individuals: The population to vaccinate. Mutated in place.
        count: Exactly how many individuals to vaccinate.
        rng: The shared random generator.
        strategy: ``"random"`` (default) or ``"high_degree"``/``"high-degree"``.
        graph: Required (non-None) pre-outbreak contact graph when
            ``strategy`` is ``"high_degree"``/``"high-degree"``.

    Returns:
        The sorted list of individual ids that were vaccinated.

    Raises:
        ValueError: For an unknown strategy, or a high-degree strategy with
            no graph available (e.g. the well-mixed contact model, which has
            no persistent network and therefore no well-defined degree).
    """
    normalized = strategy.replace("-", "_")
    if normalized == "random":
        return vaccinate_random(individuals, count, rng)
    if normalized == "high_degree":
        if graph is None:
            raise ValueError(
                "high_degree vaccination strategy requires a contact graph "
                "(the well-mixed model has no persistent network/degree).")
        return vaccinate_high_degree(individuals, count, rng, graph)
    raise ValueError(f"Unknown vaccination strategy: {strategy!r}.")


def vaccinate_cluster(
    individuals: Sequence[Individual],
    cluster_of: np.ndarray,
    cluster_rates: Sequence[float],
    rng: Generator,
    strategy: str = "random",
    graph: Optional[nx.Graph] = None,
) -> Tuple[List[int], Dict[int, int], Dict[int, int]]:
    """Vaccinate each cluster independently at its own requested rate.

    This is the cluster-specific counterpart to :func:`vaccinate`: instead of
    one global rate/count applied across the whole population, each cluster
    (as assigned by a :class:`~interaction.ClusteredContactModel`, exposed via
    its ``cluster_of`` array) gets its own vaccination count, derived from
    *that cluster's own population* -- so unequal cluster sizes are handled
    correctly instead of silently assuming equal-sized clusters. Selection
    (random or high-degree) happens strictly *within* each cluster's member
    set; nobody outside a cluster is ever a candidate for that cluster's
    count, and nobody is vaccinated twice (each individual belongs to exactly
    one cluster).

    Rounding matches the existing single-rate path
    (``Config.resolved_vaccination_count``): ``round(rate * cluster_population)``.

    Args:
        individuals: The population to vaccinate. Mutated in place.
        cluster_of: Array mapping individual id -> cluster id (as produced by
            ``ClusteredContactModel.cluster_of``).
        cluster_rates: One vaccination rate (0-1) per cluster, ordered by
            cluster id (``cluster_rates[c]`` is cluster ``c``'s rate). Must
            have exactly as many entries as there are clusters.
        rng: The shared random generator.
        strategy: ``"random"`` (default) or ``"high_degree"``/``"high-degree"``,
            applied independently within each cluster.
        graph: Required when ``strategy`` is high-degree; degree is computed
            once over the whole graph (not recomputed per cluster) and then
            restricted to each cluster's member ids.

    Returns:
        A tuple ``(vaccinated_ids, vaccinated_count_per_cluster,
        population_per_cluster)``: the sorted list of all vaccinated ids
        (across every cluster), and two dicts keyed by cluster id.

    Raises:
        ValueError: If ``cluster_of`` is missing/empty, ``cluster_rates`` has
            the wrong length, any rate is outside ``[0, 1]``, the strategy is
            unknown, high-degree is requested without a graph, or a
            cluster's requested count exceeds its eligible (susceptible)
            population.
    """
    if cluster_of is None or len(cluster_of) == 0:
        raise ValueError(
            "cluster_of is required for cluster-specific vaccination (the "
            "contact model must be 'clustered' and expose cluster membership)."
        )
    num_clusters = int(cluster_of.max()) + 1
    cluster_rates = list(cluster_rates)
    if len(cluster_rates) != num_clusters:
        raise ValueError(
            f"cluster_vaccination_rates has {len(cluster_rates)} entries but "
            f"there are {num_clusters} clusters; they must match 1:1."
        )
    for cluster_id, rate in enumerate(cluster_rates):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(
                f"cluster {cluster_id} vaccination rate {rate!r} must be in "
                "[0, 1]."
            )

    normalized = strategy.replace("-", "_")
    degree: Optional[Dict[int, int]] = None
    if normalized == "high_degree":
        if graph is None:
            raise ValueError(
                "high_degree cluster vaccination requires a contact graph."
            )
        degree = dict(graph.degree())
    elif normalized != "random":
        raise ValueError(f"Unknown vaccination strategy: {strategy!r}.")

    all_vaccinated: List[int] = []
    vaccinated_per_cluster: Dict[int, int] = {}
    population_per_cluster: Dict[int, int] = {}

    for cluster_id in range(num_clusters):
        members = [int(i) for i in np.where(cluster_of == cluster_id)[0]]
        population_per_cluster[cluster_id] = len(members)
        count = int(round(cluster_rates[cluster_id] * len(members)))
        eligible = [m for m in members if individuals[m].state is State.SUSCEPTIBLE]
        if count > len(eligible):
            raise ValueError(
                f"Cannot vaccinate {count} in cluster {cluster_id}; only "
                f"{len(eligible)} eligible (susceptible) individuals available."
            )
        if count == 0:
            vaccinated_per_cluster[cluster_id] = 0
            continue
        if normalized == "random":
            chosen = rng.choice(eligible, size=count, replace=False)
            chosen_ids = sorted(int(c) for c in chosen)
        else:
            shuffled = eligible.copy()
            rng.shuffle(shuffled)
            ranked = sorted(shuffled, key=lambda node_id: degree.get(node_id, 0),
                            reverse=True)
            chosen_ids = sorted(int(c) for c in ranked[:count])
        for individual_id in chosen_ids:
            ind = individuals[individual_id]
            ind.state = State.VACCINATED
            ind.days_in_state = 0
        vaccinated_per_cluster[cluster_id] = len(chosen_ids)
        all_vaccinated.extend(chosen_ids)

    return sorted(all_vaccinated), vaccinated_per_cluster, population_per_cluster


def vaccination_report(
    enabled: bool, strategy: str,
    vaccinated_ids: Sequence[int],
    population_size: int,
    cluster_vaccination_rates: Optional[Sequence[float]] = None,
    vaccinated_count_per_cluster: Optional[Dict[int, int]] = None,
    population_per_cluster: Optional[Dict[int, int]] = None,
) -> Dict[str, object]:
    """Summarise a (possibly empty) vaccination for statistics/reporting.

    Args:
        enabled: Whether vaccination was configured for this run.
        strategy: The vaccination strategy used (e.g. ``"random"``).
        vaccinated_ids: The ids returned by the vaccination method.
        population_size: Total population, for the coverage fraction.
        cluster_vaccination_rates: The per-cluster rates requested, if this
            was a cluster-specific run (``None`` for an ordinary uniform run).
        vaccinated_count_per_cluster: Cluster id -> number vaccinated, for a
            cluster-specific run.
        population_per_cluster: Cluster id -> cluster population, for a
            cluster-specific run.

    Returns:
        Dict with the original (unchanged) keys -- ``vaccination_enabled``,
        ``vaccination_strategy``, ``number_vaccinated``,
        ``vaccination_coverage`` -- plus ``vaccinated_count`` (alias of
        ``number_vaccinated``) and ``overall_vaccination_rate`` (alias of
        ``vaccination_coverage``, i.e. total vaccinated / total population --
        *not* the arithmetic mean of per-cluster rates, which would be wrong
        for unequal cluster sizes). When cluster metadata is supplied, also
        includes ``cluster_vaccination_rates``, ``vaccinated_count_per_cluster``,
        and ``population_per_cluster``.
    """
    number_vaccinated = len(vaccinated_ids)
    coverage = number_vaccinated / population_size if population_size else 0.0
    report: Dict[str, object] = {
        "vaccination_enabled": enabled,
        "vaccination_strategy": strategy if enabled else None,
        "number_vaccinated": number_vaccinated,
        "vaccination_coverage": coverage,
        "vaccinated_count": number_vaccinated,
        "overall_vaccination_rate": coverage,
    }
    if cluster_vaccination_rates is not None:
        report["cluster_vaccination_rates"] = list(cluster_vaccination_rates)
        report["vaccinated_count_per_cluster"] = dict(
            vaccinated_count_per_cluster or {})
        report["population_per_cluster"] = dict(population_per_cluster or {})
    return report
