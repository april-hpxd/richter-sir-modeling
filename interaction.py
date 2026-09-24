"""The interaction layer: who meets whom each day.

* :class:`WellMixedContactModel` -- any individual may meet any other
  (homogeneous-mixing assumption; used for validation). Each infectious
  person meets a fixed number of distinct others drawn uniformly.
* :class:`WattsStrogatzContactModel` -- each person's contacts are their
  neighbours in a small-world network graph.
* :class:`RandomNetworkContactModel` -- a seeded, persistent random graph in
  which each person has a random degree within configured bounds.
* :class:`DailyRandomContactModel` -- same *number* of contacts as the
  clustered model, redrawn uniformly each day, with no cluster locality.
  This is the regular-mixing control for clustering experiments.
* :class:`ClusteredContactModel` -- people belong to local clusters. Each
  day every person draws a limited number of contacts; each contact is
  taken from their own cluster with probability ``p`` and from outside
  with probability ``1-p``. Cluster membership is persistent; the people
  contacted are not.

A **contact** is one opportunity for transmission on that day: if A is
infectious and B is on A's contact list, B is exposed with the configured
infection probability. Contacts are not households, workplaces, or lasting
relationships unless a persistent-network model is used.

Daily models regenerate contacts every simulation day from the engine's RNG
stream. Persistent-network models keep the same neighbours every day.

All randomness flows through a single NumPy :class:`~numpy.random.Generator`
passed in by the caller, so contact draws stay part of the one reproducible
random stream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence

import networkx as nx
import numpy as np
from networkx.algorithms.graphical import is_graphical
from numpy.random import Generator


def draw_daily_degree(rng: Generator, min_degree: int, max_degree: int,
                     cap: int) -> int:
    """Draw how many distinct others one person meets today.

    The count is uniform on ``[min_degree, max_degree]`` inclusive, then
    clamped so nobody is asked to meet more people than exist.
    """
    lo = max(0, min(min_degree, cap))
    hi = max(lo, min(max_degree, cap))
    if hi <= 0:
        return 0
    return int(rng.integers(lo, hi + 1))


def sample_other_ids(rng: Generator, population_size: int, person_id: int,
                     k: int, exclude: Optional[set] = None) -> list[int]:
    """Sample ``k`` distinct ids other than ``person_id`` (and ``exclude``)."""
    forbidden = exclude if exclude is not None else set()
    available = [i for i in range(population_size)
                 if i != person_id and i not in forbidden]
    if not available or k <= 0:
        return []
    k = min(k, len(available))
    chosen = rng.choice(available, size=k, replace=False)
    return [int(x) for x in (chosen.tolist() if hasattr(chosen, "tolist") else [chosen])]


def sample_localized_contacts(
    rng: Generator,
    person_id: int,
    k: int,
    in_cluster: Sequence[int],
    out_cluster: Sequence[int],
    within_probability: float,
    population_size: int,
) -> list[int]:
    """Sample ``k`` distinct contacts with a per-contact locality probability.

    Each of the ``k`` slots independently tries to pick a cluster-mate with
    probability ``within_probability`` (or someone outside with probability
    ``1 - within_probability``). This is *not* "contact ``p`` of the
    population": it is "of this person's limited daily meetings, about
    ``p`` of them are local."

    If the chosen pool is empty or exhausted (tiny clusters, ``p = 1`` with
    ``k`` larger than cluster size minus one), the sampler falls back to the
    other pool, then to anyone else. Self-contact and same-day duplicates
    are never returned.
    """
    chosen: list[int] = []
    chosen_set: set[int] = set()
    in_pool = [i for i in in_cluster if i != person_id]
    out_pool = [i for i in out_cluster if i != person_id]
    k = min(k, population_size - 1)

    for _ in range(k):
        want_local = bool(rng.random() < within_probability)
        primary = in_pool if want_local else out_pool
        available = [i for i in primary if i not in chosen_set]
        if not available:
            secondary = out_pool if want_local else in_pool
            available = [i for i in secondary if i not in chosen_set]
        if not available:
            available = [i for i in range(population_size)
                         if i != person_id and i not in chosen_set]
        if not available:
            break
        pick = int(rng.choice(available))
        chosen.append(pick)
        chosen_set.add(pick)
    return chosen


def graph_from_contact_lists(population_size: int,
                             contact_lists: Sequence[Sequence[int]]) -> nx.Graph:
    """Build an undirected graph from per-person contact lists.

    An edge ``{i, j}`` is drawn if *either* person listed the other. The
    engine still transmits only along each infectious person's own list;
    the graph is the union used for visualization and clustering coefficients.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(population_size))
    for person_id, partners in enumerate(contact_lists):
        for partner in partners:
            if partner != person_id:
                graph.add_edge(int(person_id), int(partner))
    return graph


def contact_structure_stats(
    contact_lists: Sequence[Sequence[int]],
    cluster_of: Optional[np.ndarray] = None,
    graph: Optional[nx.Graph] = None,
) -> Dict[str, float]:
    """Summarise one day's (or a persistent network's) contact structure.

    These numbers are what make a clustered-vs-regular comparison
    interpretable: mean contacts should match, while the within-cluster
    proportion should not.
    """
    n = len(contact_lists)
    degrees = np.array([len(partners) for partners in contact_lists], dtype=float)
    mean_k = float(degrees.mean()) if n else 0.0
    std_k = float(degrees.std(ddof=1)) if n > 1 else 0.0

    within = between = 0
    if cluster_of is not None:
        for person_id, partners in enumerate(contact_lists):
            my_cluster = int(cluster_of[person_id])
            for partner in partners:
                if int(cluster_of[int(partner)]) == my_cluster:
                    within += 1
                else:
                    between += 1
    total_directed = within + between
    within_prop = within / total_directed if total_directed else float("nan")
    between_prop = between / total_directed if total_directed else float("nan")

    if graph is None:
        graph = graph_from_contact_lists(n, contact_lists)
    avg_degree = (
        float(np.mean([d for _, d in graph.degree()])) if n else 0.0)
    clustering = float(nx.average_clustering(graph)) if n > 1 else 0.0

    num_clusters = 0.0
    mean_cluster_size = float("nan")
    if cluster_of is not None and n:
        num_clusters = float(int(cluster_of.max()) + 1)
        mean_cluster_size = n / num_clusters if num_clusters else float("nan")

    return {
        "population_size": float(n),
        "average_contacts": mean_k,
        "std_contacts": std_k,
        "within_cluster_contact_proportion": within_prop,
        "between_cluster_contact_proportion": between_prop,
        "num_clusters": num_clusters,
        "mean_cluster_size": mean_cluster_size,
        "average_degree": avg_degree,
        "clustering_coefficient": clustering,
    }


def network_topology_report(
    graph: nx.Graph, path_length_node_cap: int = 500,
) -> Dict[str, object]:
    """One-off structural report for a contact graph.

    Unlike :func:`contact_structure_stats` (designed to be called once per
    simulated day, then averaged over many days), this is meant to be called
    once per network -- e.g. at experiment setup -- to confirm the network
    actually has the structural properties an experiment design assumes:
    matched mean degree between two scenarios, a network that is (or isn't)
    fully connected, etc.

    Mean shortest-path length is expensive (roughly O(n * m) via
    breadth-first search from every node) and undefined for a disconnected
    graph, so it is computed on the largest connected component only, and
    skipped entirely above ``path_length_node_cap`` nodes. ``path_length_note``
    always explains what (if anything) was skipped and why.
    """
    n = graph.number_of_nodes()
    degrees = np.array([d for _, d in graph.degree()], dtype=float)
    components = list(nx.connected_components(graph))
    num_components = len(components)
    largest_component_size = max((len(c) for c in components), default=0)

    average_shortest_path_length = float("nan")
    path_length_computed = False
    if n == 0:
        path_length_note = "empty graph"
    elif n > path_length_node_cap:
        path_length_note = (
            f"skipped: {n} nodes exceeds path_length_node_cap={path_length_node_cap}"
        )
    elif largest_component_size < 2:
        path_length_note = "skipped: no component with more than one node"
    else:
        if num_components == 1:
            target_graph = graph
            path_length_note = "computed on the full (connected) graph"
        else:
            largest_nodes = max(components, key=len)
            target_graph = graph.subgraph(largest_nodes)
            path_length_note = (
                f"computed on the largest connected component only "
                f"({largest_component_size}/{n} nodes); graph has "
                f"{num_components} components"
            )
        average_shortest_path_length = float(
            nx.average_shortest_path_length(target_graph))
        path_length_computed = True

    return {
        "node_count": float(n),
        "edge_count": float(graph.number_of_edges()),
        "mean_degree": float(degrees.mean()) if n else 0.0,
        "std_degree": float(degrees.std(ddof=1)) if n > 1 else 0.0,
        "min_degree": float(degrees.min()) if n else 0.0,
        "max_degree": float(degrees.max()) if n else 0.0,
        "clustering_coefficient": float(nx.average_clustering(graph)) if n > 1 else 0.0,
        "num_connected_components": float(num_components),
        "largest_component_size": float(largest_component_size),
        "largest_component_fraction": (
            float(largest_component_size) / n if n else float("nan")),
        "average_shortest_path_length": average_shortest_path_length,
        "path_length_computed": path_length_computed,
        "path_length_note": path_length_note,
    }


class ContactModel(ABC):
    """Abstract interface for a daily contact structure.

    A contact model answers one question: given an individual, which other
    individuals do they interact with on a given day? Concrete subclasses
    define the population structure (well-mixed, networked, spatial, ...)
    without the engine needing to know which.
    """

    def prepare_day(self, rng: Generator) -> None:
        """Optional: regenerate today's contacts for the whole population.

        Persistent-network models leave this as a no-op. Daily models must
        draw from ``rng`` only (the simulation's stream).
        """

    @abstractmethod
    def contacts(self, individual_id: int, rng: Generator) -> np.ndarray:
        """Return the ids that ``individual_id`` interacts with today.

        Args:
            individual_id: The id of the (infectious) individual seeking
                contacts.
            rng: The shared random generator; any stochastic contact model must
                draw from this and only this, to preserve reproducibility.

        Returns:
            A 1-D array of *other* individual ids (never including
            ``individual_id`` itself). Ids may be susceptible or not; the
            engine decides what happens on each contact.
        """
        raise NotImplementedError


class WellMixedContactModel(ContactModel):
    """Homogeneous mixing: contacts are uniformly random other individuals.

    Each day an individual meets ``daily_contacts`` distinct other people drawn
    uniformly at random from the whole population. This is the deliberately
    simple, temporary stand-in for a real social network, used here to validate
    the disease dynamics in isolation.

    Attributes:
        population_size: Total number of individuals.
        daily_contacts: Number of distinct others each individual meets daily.
    """

    def __init__(self, population_size: int, daily_contacts: int) -> None:
        """Initialise the well-mixed model.

        Args:
            population_size: Total number of individuals (>= 2).
            daily_contacts: Distinct contacts per individual per day; clamped
                to at most ``population_size - 1`` (you cannot meet more than
                everyone else).
        """
        self.population_size = population_size
        self.daily_contacts = min(daily_contacts, population_size - 1)

    def contacts(self, individual_id: int, rng: Generator) -> np.ndarray:
        """Sample ``daily_contacts`` distinct other individuals uniformly.

        Self-contact is excluded by sampling ``daily_contacts`` positions from
        the ``population_size - 1`` *other* individuals and mapping any position
        at or beyond ``individual_id`` up by one. This is exact (no rejection
        loop) and consumes the random stream deterministically.

        Args:
            individual_id: The individual seeking contacts.
            rng: The shared random generator.

        Returns:
            A 1-D array of distinct other individual ids.
        """
        # Draw distinct positions in the reduced index space [0, n-2].
        positions = rng.choice(
            self.population_size - 1,
            size=self.daily_contacts,
            replace=False,
        )
        # Map positions >= individual_id up by one to skip the individual.
        return np.where(positions >= individual_id, positions + 1, positions)


class WattsStrogatzContactModel(ContactModel):
    """Small-world network: contacts are graph neighbours.

    Each person's daily contacts are their neighbours in a Watts-Strogatz
    small-world network. The network is built once at construction (deterministic
    via an RNG seed) and remains fixed across all simulation days. On each day,
    an infectious individual has daily contact with their graph neighbours
    (possibly augmented by random rewiring to introduce long-range edges).

    Attributes:
        graph: The underlying networkx small-world graph (node ids = individual ids).
    """

    def __init__(self, population_size: int, k: int = 4, p: float = 0.1,
                 rng: Generator | None = None) -> None:
        """Build a Watts-Strogatz small-world network.

        Args:
            population_size: Number of nodes (individuals).
            k: Each node connected to k nearest neighbours (on each side).
            p: Rewiring probability for small-world edges (0 = lattice, 1 = random).
            rng: Optional NumPy generator for reproducible graph construction.
                If not given, a new default generator is used.
        """
        if rng is None:
            rng = np.random.default_rng()

        seed = int(rng.integers(0, 2**31))
        self.graph = nx.watts_strogatz_graph(
            n=population_size,
            k=k,
            p=p,
            seed=seed,
        )

    def contacts(self, individual_id: int, rng: Generator) -> np.ndarray:
        """Return the graph neighbours of this individual.

        The RNG parameter is unused here (the network is static), but accepted
        to maintain the :class:`ContactModel` interface.

        Args:
            individual_id: The individual whose contacts we need.
            rng: The shared random generator (unused; network is deterministic).

        Returns:
            A 1-D array of this node's neighbours in the graph.
        """
        neighbors = list(self.graph.neighbors(individual_id))
        return np.array(neighbors, dtype=np.int64)


class RandomNetworkContactModel(ContactModel):
    """A seeded undirected social graph with bounded random node degrees.

    Each resident is assigned a degree drawn uniformly from ``min_degree`` to
    ``max_degree`` (inclusive), subject to the graph being feasible.  A simple
    graph with exactly those degrees is then built and rewired, so the visible
    links are persistent social contacts rather than new random dots each day.
    """

    def __init__(self, population_size: int, min_degree: int = 1,
                 max_degree: int = 7, rng: Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        if population_size < 2:
            raise ValueError("population_size must be >= 2.")

        self.population_size = population_size
        self.min_degree = min(min_degree, population_size - 1)
        self.max_degree = min(max_degree, population_size - 1)
        if self.min_degree < 1 or self.max_degree < self.min_degree:
            raise ValueError("Invalid random-network degree bounds.")

        degrees = self._graphical_degree_sequence(rng)
        self.graph = nx.havel_hakimi_graph(degrees)

        # Havel-Hakimi provides the exact requested degrees.  Degree-preserving
        # swaps remove its construction-order bias while keeping the run seeded.
        swaps = max(1, self.graph.number_of_edges() * 3)
        try:
            nx.double_edge_swap(
                self.graph, nswap=swaps, max_tries=swaps * 20,
                seed=int(rng.integers(0, 2**31)),
            )
        except (nx.NetworkXAlgorithmError, nx.NetworkXError):
            # A very small/dense graph may not admit enough swaps; its valid
            # Havel-Hakimi graph still has the promised degree for every node.
            pass

    def _graphical_degree_sequence(self, rng: Generator) -> list[int]:
        """Draw a feasible degree for every node without relaxing the bounds."""
        for _ in range(1_000):
            degrees = rng.integers(
                self.min_degree, self.max_degree + 1,
                size=self.population_size,
            ).tolist()
            if sum(degrees) % 2:
                adjustable = [
                    i for i, degree in enumerate(degrees)
                    if degree < self.max_degree or degree > self.min_degree
                ]
                index = int(rng.choice(adjustable))
                degrees[index] += 1 if degrees[index] < self.max_degree else -1
            if is_graphical(degrees):
                return degrees
        raise RuntimeError("Could not generate a graphical random degree sequence.")

    def contacts(self, individual_id: int, rng: Generator) -> np.ndarray:
        """Return this person's persistent graph neighbours."""
        return np.fromiter(self.graph.neighbors(individual_id), dtype=np.int64)


class DailyRandomContactModel(ContactModel):
    """Homogeneous mixing with a bounded, resampled daily contact count.

    This is the regular-population control for clustering experiments: the
    same per-person daily contact *rate* as :class:`ClusteredContactModel`,
    but partners are drawn uniformly from the rest of the population. Cluster
    locality is the only systematic difference when the two models share
    ``min_degree`` / ``max_degree``.
    """

    def __init__(self, population_size: int, min_degree: int = 1,
                 max_degree: int = 7, rng: Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        if population_size < 2:
            raise ValueError("population_size must be >= 2.")
        cap = population_size - 1
        self.population_size = population_size
        self.min_degree = max(0, min(min_degree, cap))
        self.max_degree = max(self.min_degree, min(max_degree, cap))
        self.contact_lists: list[list[int]] = [[] for _ in range(population_size)]
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(population_size))
        self.last_stats: Dict[str, float] = {}
        self._accumulated: list[Dict[str, float]] = []
        self.prepare_day(rng)

    def prepare_day(self, rng: Generator) -> None:
        """Redraw every person's contacts uniformly from the population."""
        cap = self.population_size - 1
        lists: list[list[int]] = []
        for person_id in range(self.population_size):
            k = draw_daily_degree(rng, self.min_degree, self.max_degree, cap)
            lists.append(sample_other_ids(rng, self.population_size, person_id, k))
        self.contact_lists = lists
        self.graph = graph_from_contact_lists(self.population_size, lists)
        stats = contact_structure_stats(lists, cluster_of=None, graph=self.graph)
        self.last_stats = stats
        self._accumulated.append(stats)

    def contacts(self, individual_id: int, rng: Generator) -> np.ndarray:
        """Return this person's contacts from the current day (RNG unused)."""
        return np.asarray(self.contact_lists[individual_id], dtype=np.int64)

    def mean_contact_stats(self) -> Dict[str, float]:
        """Average :func:`contact_structure_stats` over all prepared days."""
        return _mean_stat_dicts(self._accumulated)


class ClusteredContactModel(ContactModel):
    """Daily contacts with configurable within-cluster locality.

    Cluster *membership* is assigned once (even split of the population,
    shuffled with the constructor RNG) and then held fixed. Contact *partners*
    are redrawn every day:

    1. Person ``i`` draws a contact count ``k`` uniformly from
       ``[min_degree, max_degree]`` (the same bounds used by the regular
       random-network / daily-random models, so mean degree is matched).
    2. Each of those ``k`` meetings independently comes from ``i``'s own
       cluster with probability ``within_cluster_contact_probability`` and
       from the rest of the population with probability ``1 - p``.

    ``random_chance`` is kept as the complementary parameter
    ``1 - within_cluster_contact_probability`` so older configs still work.

    This design avoids the previous confounder, where clustered cities were
    given a dense intra-cluster ring (often near-complete for small clusters)
    and therefore many more contacts than a 1–7 random network.

    Attributes:
        graph: Undirected union of today's per-person contact lists.
        cluster_of: Array mapping individual id -> cluster id.
        clusters: Per-cluster list of member individual ids.
        within_cluster_contact_probability: Per-contact locality ``p``.
        random_chance: ``1 - p`` (legacy name for between-cluster chance).
    """

    def __init__(self, population_size: int, num_clusters: int = 4,
                 random_chance: Optional[float] = None, daily_contacts: int = 8,
                 rng: Generator | None = None,
                 within_cluster_contact_probability: Optional[float] = None,
                 min_degree: Optional[int] = None,
                 max_degree: Optional[int] = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        if population_size < 2:
            raise ValueError("population_size must be >= 2.")
        if num_clusters < 1:
            raise ValueError("num_clusters must be >= 1.")
        if num_clusters > population_size:
            raise ValueError("num_clusters must not exceed population_size.")

        if (within_cluster_contact_probability is not None
                and random_chance is not None):
            expected = 1.0 - random_chance
            if abs(within_cluster_contact_probability - expected) > 1e-9:
                raise ValueError(
                    "within_cluster_contact_probability must equal "
                    "1 - random_chance when both are set.")
        if within_cluster_contact_probability is None:
            chance = 0.1 if random_chance is None else random_chance
            within_cluster_contact_probability = 1.0 - chance
        if not 0.0 <= within_cluster_contact_probability <= 1.0:
            raise ValueError(
                "within_cluster_contact_probability must be in [0, 1].")

        cap = population_size - 1
        if min_degree is None and max_degree is None:
            # Matched to the project's default random-network bounds so
            # clustered vs regular does not change contact *quantity*.
            min_degree, max_degree = 1, 7
        elif min_degree is None:
            min_degree = max_degree if max_degree is not None else daily_contacts
        elif max_degree is None:
            max_degree = min_degree

        self.population_size = population_size
        self.num_clusters = num_clusters
        self.within_cluster_contact_probability = float(
            within_cluster_contact_probability)
        self.random_chance = 1.0 - self.within_cluster_contact_probability
        self.min_degree = max(0, min(int(min_degree), cap))
        self.max_degree = max(self.min_degree, min(int(max_degree), cap))

        order = rng.permutation(population_size)
        groups = np.array_split(order, num_clusters)
        self.cluster_of = np.empty(population_size, dtype=np.int64)
        self.clusters: list[list[int]] = []
        for cluster_id, members in enumerate(groups):
            self.cluster_of[members] = cluster_id
            self.clusters.append(sorted(int(m) for m in members))

        self.contact_lists: list[list[int]] = [[] for _ in range(population_size)]
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(population_size))
        self.last_stats: Dict[str, float] = {}
        self._accumulated: list[Dict[str, float]] = []
        self.prepare_day(rng)

    def prepare_day(self, rng: Generator) -> None:
        """Redraw today's contacts, respecting cluster locality probability."""
        cap = self.population_size - 1
        lists: list[list[int]] = []
        p = self.within_cluster_contact_probability
        for person_id in range(self.population_size):
            my_cluster = int(self.cluster_of[person_id])
            in_cluster = self.clusters[my_cluster]
            out_cluster = [j for j in range(self.population_size)
                           if int(self.cluster_of[j]) != my_cluster]
            k = draw_daily_degree(rng, self.min_degree, self.max_degree, cap)
            lists.append(sample_localized_contacts(
                rng, person_id, k, in_cluster, out_cluster, p,
                self.population_size))
        self.contact_lists = lists
        self.graph = graph_from_contact_lists(self.population_size, lists)
        stats = contact_structure_stats(lists, self.cluster_of, self.graph)
        self.last_stats = stats
        self._accumulated.append(stats)

    def contacts(self, individual_id: int, rng: Generator) -> np.ndarray:
        """Return this person's contacts from the current day (RNG unused)."""
        return np.asarray(self.contact_lists[individual_id], dtype=np.int64)

    def cross_cluster_edge_fraction(self) -> float:
        """Return the fraction of *today's* undirected edges that cross clusters."""
        total = self.graph.number_of_edges()
        if total == 0:
            return 0.0
        cross = sum(1 for u, v in self.graph.edges()
                   if self.cluster_of[u] != self.cluster_of[v])
        return cross / total

    def mean_contact_stats(self) -> Dict[str, float]:
        """Average :func:`contact_structure_stats` over all prepared days."""
        return _mean_stat_dicts(self._accumulated)


def _mean_stat_dicts(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    out: Dict[str, float] = {}
    for key in keys:
        values = [row[key] for row in rows if not np.isnan(row[key])]
        out[key] = float(np.mean(values)) if values else float("nan")
    return out


def persistent_contact_lists(graph: nx.Graph, population_size: int) -> list[list[int]]:
    """Neighbour lists for a persistent undirected graph."""
    return [list(graph.neighbors(i)) for i in range(population_size)]


def build_contact_model(model_type: str, population_size: int, config,
                        rng: Generator) -> ContactModel:
    """Construct the named contact model from a :class:`~config.Config`.

    Shared by single-city and regional builders so city 0 in a one-city
    regional run uses the same constructor arguments as a standalone run.
    """
    if hasattr(config, "contact_degree_bounds"):
        min_degree, max_degree = config.contact_degree_bounds(population_size)
    else:
        cap = max(1, population_size - 1)
        min_degree = max(1, min(int(config.daily_contacts_min), cap))
        max_degree = max(min_degree, min(int(config.daily_contacts_max), cap))
    if model_type == "random-network":
        return RandomNetworkContactModel(
            population_size=population_size,
            min_degree=config.random_degree_min,
            max_degree=config.random_degree_max,
            rng=rng,
        )
    if model_type == "well-mixed":
        return WellMixedContactModel(
            population_size=population_size,
            daily_contacts=min(config.daily_contacts, population_size - 1),
        )
    if model_type == "watts-strogatz":
        return WattsStrogatzContactModel(
            population_size=population_size,
            k=config.watts_strogatz_k,
            p=config.watts_strogatz_p,
            rng=rng,
        )
    if model_type == "daily-random":
        return DailyRandomContactModel(
            population_size=population_size,
            min_degree=min_degree,
            max_degree=max_degree,
            rng=rng,
        )
    if model_type == "clustered":
        p = config.within_cluster_contact_probability
        if p is None:
            p = 1.0 - config.random_chance
        return ClusteredContactModel(
            population_size=population_size,
            num_clusters=min(config.num_clusters, population_size),
            within_cluster_contact_probability=p,
            min_degree=min_degree,
            max_degree=max_degree,
            rng=rng,
        )
    raise ValueError(f"Unknown contact model: {model_type}")


