"""Contact-network construction for the city model.

Instead of the classical homogeneous-mixing SIR model (where every individual
can infect every other), we model the population as a *contact network*: each
node is one person and each edge is a recurring social contact. Disease can
only travel along edges. This captures the intuition that people mostly
interact within small, overlapping communities (households, schools,
workplaces, neighbourhoods) and only occasionally reach outside them.

We use the **Watts-Strogatz small-world** model for exactly this reason. It
starts from a ring lattice -- each person connected to their ``k`` nearest
neighbours, i.e. dense *local* clustering -- and then rewires a fraction of
edges at random, inserting a few long-range "shortcuts". The result reproduces
the two hallmarks of real social networks: high local clustering *and* short
average path length between any two people.

This module builds the graph and also assigns each node a stable 2-D position.
Fixing positions once (here) is what lets the animation stay *spatially
consistent* across frames -- the layout never jumps around as the disease
spreads.
"""

from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import numpy as np

from config import Config

# A node position is an (x, y) coordinate.
Position = Tuple[float, float]


def build_contact_network(config: Config) -> nx.Graph:
    """Build a Watts-Strogatz small-world contact network.

    Args:
        config: Simulation configuration. Uses ``population_size`` (number of
            nodes), ``average_contacts`` (the Watts-Strogatz ``k``),
            ``rewiring_probability`` (``beta``), and ``random_seed``.

    Returns:
        An undirected :class:`networkx.Graph` with ``population_size`` nodes
        labelled ``0 .. population_size - 1``. The graph carries two graph-level
        attributes for downstream use:

        * ``graph["config"]`` -- the originating :class:`Config`.
        * ``graph["community"]`` -- a dict mapping each node to an integer
          community id (contiguous blocks of the underlying ring), used by the
          geographic visualization to lay out neighbourhoods.

    Notes:
        The Watts-Strogatz construction can occasionally produce a disconnected
        graph. That is epidemiologically meaningful (an outbreak may not reach
        an isolated cluster), so we deliberately do **not** force connectivity;
        we only record it via :func:`network_summary`.
    """
    n = config.population_size
    k = config.average_contacts

    graph = nx.watts_strogatz_graph(
        n=n,
        k=k,
        p=config.rewiring_probability,
        seed=config.random_seed,
    )

    # Attach a stable spatial layout so every visualization frame agrees.
    positions = compute_layout(graph, config)
    nx.set_node_attributes(graph, positions, name="pos")

    # Record community structure derived from the ring: contiguous index
    # blocks behave like local neighbourhoods before rewiring. This is a cheap,
    # deterministic proxy that the geographic view uses to cluster households.
    communities = assign_communities(graph, config)
    nx.set_node_attributes(graph, communities, name="community")

    # Stash context on the graph object for convenience / future City reuse.
    graph.graph["config"] = config
    graph.graph["community"] = communities
    return graph


def compute_layout(graph: nx.Graph, config: Config) -> Dict[int, Position]:
    """Compute a fixed 2-D position for every node.

    A force-directed (spring) layout places well-connected clusters near one
    another, which makes small-world structure visible: tight neighbourhoods
    form visual clumps joined by the occasional long shortcut edge.

    The layout is computed once and reused for the entire animation so the
    graph does not visually "jump" between frames.

    Args:
        graph: The contact network.
        config: Configuration (used only for ``random_seed`` so the layout is
            reproducible).

    Returns:
        Mapping from node id to an ``(x, y)`` tuple.
    """
    # spring_layout is O(n^2) per iteration; for large cities we reduce the
    # iteration count so layout time stays reasonable. The disease dynamics are
    # unaffected -- this only positions dots on a screen.
    n = graph.number_of_nodes()
    iterations = 50 if n <= 2000 else 20

    raw = nx.spring_layout(
        graph,
        seed=config.random_seed,
        iterations=iterations,
    )
    # Convert numpy arrays to plain float tuples for stable, hashable storage.
    return {node: (float(xy[0]), float(xy[1])) for node, xy in raw.items()}


def assign_communities(graph: nx.Graph, config: Config) -> Dict[int, int]:
    """Assign each node to a community id based on contiguous ring blocks.

    In the Watts-Strogatz ring, nodes with nearby indices start out densely
    interconnected -- a natural stand-in for a neighbourhood, school, or
    workplace before random rewiring. We partition the index range into a
    handful of equal blocks and label each node with its block id.

    The SIR dynamics never use it. Just for geographic simulation

    Args:
        graph: The contact network.
        config: Configuration (uses ``population_size``).

    Returns:
        Mapping from node id to an integer community id.
    """
    n = config.population_size
    # Aim for communities of ~50 people, bounded to a sensible range so both
    # tiny test cities and large cities look reasonable.
    num_communities = max(1, min(24, round(n / 50)))
    block_size = n / num_communities
    return {node: min(num_communities - 1, int(node / block_size))
            for node in graph.nodes()}


def network_summary(graph: nx.Graph) -> Dict[str, float]:
    """Compute descriptive statistics about the generated network.

    These are useful for a run header ("is this actually small-world?") and for
    the eventual write-up, but they do not affect the simulation.

    Args:
        graph: The contact network.

    Returns:
        Dictionary with node/edge counts, mean degree, average clustering
        coefficient, connectivity flag, and (for connected graphs) the average
        shortest-path length.
    """

