"""The interaction layer: who meets whom each day.

This is the single seam that the whole architecture is designed around. The
disease engine never decides *how* people come into contact; it only asks a
:class:`ContactModel` "who does person ``i`` interact with today?" and applies
transmission along whatever ids come back.

Three implementations ship here:

* :class:`WellMixedContactModel` -- any individual may meet any other
  (homogeneous-mixing assumption; used for validation).
* :class:`WattsStrogatzContactModel` -- each person's contacts are their
  neighbours in a small-world network graph.
* :class:`RandomNetworkContactModel` -- a seeded, persistent random graph in
  which each person has a random degree within configured bounds.

All randomness flows through a single NumPy :class:`~numpy.random.Generator`
passed in by the caller, so contact draws stay part of the one reproducible
random stream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
from networkx.algorithms.graphical import is_graphical
from numpy.random import Generator


class ContactModel(ABC):
    """Abstract interface for a daily contact structure.

    A contact model answers one question: given an individual, which other
    individuals do they interact with on a given day? Concrete subclasses
    define the population structure (well-mixed, networked, spatial, ...)
    without the engine needing to know which.
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
        except nx.NetworkXAlgorithmError:
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
