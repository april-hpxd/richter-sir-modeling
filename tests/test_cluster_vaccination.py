"""Tests for optional cluster-specific vaccination (purely additive).

These cover the new ``vaccinate_cluster`` function in ``vaccination.py``, the
``Config.cluster_vaccination_rates`` field/validation, and the
``Simulation``-level wiring, while confirming the existing uniform
``--vaccination-rate`` path is completely unaffected when the new option is
not used. See CHANGELOG_RESEARCH.md for the research write-up.
"""

import networkx as nx
import numpy as np
import pytest

from config import Config
from disease_model import Individual, State
from simulation import Simulation
from vaccination import vaccinate_cluster


def _population(n):
    return [Individual(id=i) for i in range(n)]


def _cluster_of(sizes):
    """Build a cluster_of array for clusters of the given sizes, in order."""
    cluster_of = np.empty(sum(sizes), dtype=np.int64)
    start = 0
    for cluster_id, size in enumerate(sizes):
        cluster_of[start:start + size] = cluster_id
        start += size
    return cluster_of


# ---------------------------------------------------------------------
# vaccinate_cluster: direct unit tests
# ---------------------------------------------------------------------

def test_per_cluster_counts_correct():
    cluster_of = _cluster_of([100, 100, 100])
    individuals = _population(300)
    vaccinated, per_cluster, population_per_cluster = vaccinate_cluster(
        individuals, cluster_of, [0.0, 0.2, 0.6], np.random.default_rng(1))
    assert population_per_cluster == {0: 100, 1: 100, 2: 100}
    assert per_cluster == {0: 0, 1: 20, 2: 60}
    assert len(vaccinated) == 80
    assert sum(1 for ind in individuals if ind.state is State.VACCINATED) == 80


def test_overall_coverage_is_weighted_not_mean_of_rates():
    cluster_of = _cluster_of([50, 100, 200])
    individuals = _population(350)
    vaccinated, per_cluster, population_per_cluster = vaccinate_cluster(
        individuals, cluster_of, [0.0, 0.2, 0.6], np.random.default_rng(2))
    assert per_cluster == {0: 0, 1: 20, 2: 120}
    overall = len(vaccinated) / sum(population_per_cluster.values())
    simple_mean = np.mean([0.0, 0.2, 0.6])
    assert overall == pytest.approx(140 / 350)
    assert overall != pytest.approx(simple_mean)


def test_no_cross_cluster_vaccination():
    cluster_of = _cluster_of([20, 20])
    individuals = _population(40)
    vaccinated, _, _ = vaccinate_cluster(
        individuals, cluster_of, [1.0, 0.0], np.random.default_rng(3))
    # Cluster 0 is entirely vaccinated, cluster 1 entirely untouched.
    assert set(vaccinated) == set(range(20))


def test_no_duplicates_across_clusters():
    cluster_of = _cluster_of([30, 30, 30])
    individuals = _population(90)
    vaccinated, _, _ = vaccinate_cluster(
        individuals, cluster_of, [0.5, 0.5, 0.5], np.random.default_rng(4))
    assert len(vaccinated) == len(set(vaccinated))


def test_zero_and_full_rate_clusters():
    cluster_of = _cluster_of([10, 10])
    individuals = _population(20)
    vaccinated, per_cluster, _ = vaccinate_cluster(
        individuals, cluster_of, [0.0, 1.0], np.random.default_rng(5))
    assert per_cluster[0] == 0
    assert per_cluster[1] == 10
    assert set(vaccinated) == set(range(10, 20))


def test_random_strategy_reproducible():
    cluster_of = _cluster_of([50, 50])
    chosen_a, _, _ = vaccinate_cluster(
        _population(100), cluster_of, [0.3, 0.3], np.random.default_rng(42))
    chosen_b, _, _ = vaccinate_cluster(
        _population(100), cluster_of, [0.3, 0.3], np.random.default_rng(42))
    assert chosen_a == chosen_b


def test_high_degree_within_cluster_picks_highest_degree_nodes():
    # Two disjoint cliques of different "hubs": build a graph where cluster 0
    # nodes 0-9 have degree correlated with id (node 9 highest), so the
    # highest-degree pick within cluster 0 should favor high-id nodes.
    graph = nx.Graph()
    graph.add_nodes_from(range(20))
    # cluster 0: 0..9, star-like increasing degree; cluster 1: 10..19 regular
    for i in range(10):
        for j in range(i + 1, 10):
            if j - i <= (i % 5) + 1:
                graph.add_edge(i, j)
    for i in range(10, 20):
        graph.add_edge(i, 10 + (i + 1 - 10) % 10)

    cluster_of = _cluster_of([10, 10])
    degree = dict(graph.degree())
    individuals = _population(20)
    vaccinated, per_cluster, _ = vaccinate_cluster(
        individuals, cluster_of, [0.3, 0.0], np.random.default_rng(6),
        strategy="high_degree", graph=graph)
    assert per_cluster[0] == 3
    cluster0_degrees = sorted(
        (degree[n] for n in range(10)), reverse=True)[:3]
    chosen_degrees = sorted(degree[n] for n in vaccinated)
    assert chosen_degrees == sorted(cluster0_degrees)


def test_invalid_rate_count_raises():
    cluster_of = _cluster_of([10, 10, 10])
    with pytest.raises(ValueError):
        vaccinate_cluster(_population(30), cluster_of, [0.1, 0.2],
                          np.random.default_rng(1))


def test_negative_rate_raises():
    cluster_of = _cluster_of([10, 10])
    with pytest.raises(ValueError):
        vaccinate_cluster(_population(20), cluster_of, [-0.1, 0.2],
                          np.random.default_rng(1))


def test_rate_above_one_raises():
    cluster_of = _cluster_of([10, 10])
    with pytest.raises(ValueError):
        vaccinate_cluster(_population(20), cluster_of, [0.1, 1.5],
                          np.random.default_rng(1))


def test_missing_cluster_of_raises():
    with pytest.raises(ValueError):
        vaccinate_cluster(_population(20), None, [0.1, 0.2],
                          np.random.default_rng(1))


# ---------------------------------------------------------------------
# Config-level validation
# ---------------------------------------------------------------------

def test_config_rejects_wrong_count_of_rates():
    with pytest.raises(ValueError):
        Config(population_size=300, contact_model="clustered", num_clusters=3,
              cluster_vaccination_rates=(0.1, 0.2))


def test_config_rejects_negative_rate():
    with pytest.raises(ValueError):
        Config(population_size=300, contact_model="clustered", num_clusters=3,
              cluster_vaccination_rates=(-0.1, 0.2, 0.3))


def test_config_rejects_rate_above_one():
    with pytest.raises(ValueError):
        Config(population_size=300, contact_model="clustered", num_clusters=3,
              cluster_vaccination_rates=(0.1, 0.2, 1.5))


def test_config_rejects_cluster_rates_without_clustered_model():
    with pytest.raises(ValueError):
        Config(population_size=300, contact_model="random-network",
              cluster_vaccination_rates=(0.1, 0.2, 0.3, 0.4))


# ---------------------------------------------------------------------
# Simulation-level integration
# ---------------------------------------------------------------------

def _clustered_config(rates=None, strategy="random", population=300,
                      num_clusters=3, seed=7):
    return Config(population_size=population, contact_model="clustered",
                 num_clusters=num_clusters, infection_probability=0.05,
                 initial_infected=2, simulation_days=5, random_seed=seed,
                 vaccination_strategy=strategy,
                 cluster_vaccination_rates=rates)


def test_simulation_applies_cluster_vaccination():
    config = _clustered_config(rates=(0.0, 0.2, 0.6))
    sim = Simulation(config)
    report = sim.vaccination_report()
    assert report["cluster_vaccination_rates"] == [0.0, 0.2, 0.6]
    assert sum(report["population_per_cluster"].values()) == 300
    assert report["vaccinated_count"] == sum(
        report["vaccinated_count_per_cluster"].values())
    # Overall rate is the weighted total, not the mean of [0, 0.2, 0.6].
    assert report["overall_vaccination_rate"] == pytest.approx(
        report["vaccinated_count"] / 300)


def test_simulation_cluster_high_degree_strategy_runs():
    config = _clustered_config(rates=(0.1, 0.1, 0.1), strategy="high_degree")
    sim = Simulation(config)
    sim.run()
    assert len(sim.vaccinated_ids) > 0


def test_uniform_vaccination_unchanged_when_option_not_used():
    """The ordinary --vaccination-rate path must be bit-for-bit identical."""
    config_old_style = Config(population_size=500, contact_model="random-network",
                              random_degree_min=2, random_degree_max=10,
                              infection_probability=0.05, initial_infected=2,
                              simulation_days=15, random_seed=99,
                              vaccination_rate=0.2, vaccination_strategy="random")
    sim = Simulation(config_old_style)
    sim.run()
    report = sim.vaccination_report()
    assert report.get("cluster_vaccination_rates") is None
    assert "vaccination_enabled" in report
    assert "vaccination_coverage" in report
    assert len(sim.vaccinated_ids) == 100
    assert sim.cluster_vaccinated_count is None
    assert sim.cluster_population is None
