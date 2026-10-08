"""Tests for Milestone-2 additions: high-degree vaccination, the
population-aware initial-infection policy, and a basic performance smoke
test. See CHANGELOG_RESEARCH.md for the research write-up these back.
"""

import time

import networkx as nx
import numpy as np
import pytest

from config import Config, population_aware_initial_infected
from disease_model import Individual, State
from simulation import Simulation
from vaccination import vaccinate, vaccinate_high_degree, vaccinate_random


def _population(n):
    return [Individual(id=i) for i in range(n)]


# ---------------------------------------------------------------------
# Part 1/2: high-degree vaccination mechanics
# ---------------------------------------------------------------------

def test_high_degree_selects_exact_count_and_no_duplicates():
    graph = nx.barabasi_albert_graph(100, 3, seed=1)
    individuals = _population(100)
    chosen = vaccinate_high_degree(individuals, 20, np.random.default_rng(1), graph)
    assert len(chosen) == 20
    assert len(chosen) == len(set(chosen))
    assert sum(1 for ind in individuals if ind.state is State.VACCINATED) == 20


def test_high_degree_picks_higher_average_degree_than_random():
    graph = nx.barabasi_albert_graph(300, 3, seed=2)
    degree = dict(graph.degree())

    random_pop = _population(300)
    random_chosen = vaccinate_random(random_pop, 60, np.random.default_rng(5))

    hd_pop = _population(300)
    hd_chosen = vaccinate_high_degree(hd_pop, 60, np.random.default_rng(5), graph)

    avg_random = np.mean([degree[i] for i in random_chosen])
    avg_hd = np.mean([degree[i] for i in hd_chosen])
    assert avg_hd > avg_random


def test_high_degree_reproducible_given_same_seed():
    graph = nx.barabasi_albert_graph(150, 3, seed=3)
    chosen_a = vaccinate_high_degree(_population(150), 30,
                                     np.random.default_rng(99), graph)
    chosen_b = vaccinate_high_degree(_population(150), 30,
                                     np.random.default_rng(99), graph)
    assert chosen_a == chosen_b


def test_high_degree_tie_break_is_seeded_not_insertion_order():
    # Regular graph: every node has the same degree, so the whole eligible
    # pool is "tied" and the selection must come from the seeded shuffle.
    graph = nx.random_regular_graph(4, 60, seed=4)
    chosen_a = vaccinate_high_degree(_population(60), 10,
                                     np.random.default_rng(1), graph)
    chosen_b = vaccinate_high_degree(_population(60), 10,
                                     np.random.default_rng(2), graph)
    assert chosen_a != chosen_b  # different seeds -> different tie-break


def test_vaccinate_dispatcher_random_matches_direct_call():
    pop_a = _population(50)
    pop_b = _population(50)
    chosen_a = vaccinate_random(pop_a, 10, np.random.default_rng(7))
    chosen_b = vaccinate(pop_b, 10, np.random.default_rng(7), strategy="random")
    assert chosen_a == chosen_b


def test_vaccinate_dispatcher_high_degree_requires_graph():
    with pytest.raises(ValueError):
        vaccinate(_population(10), 2, np.random.default_rng(1),
                 strategy="high_degree", graph=None)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        vaccinate(_population(10), 2, np.random.default_rng(1), strategy="bogus")


# ---------------------------------------------------------------------
# Part 6: strategy comparison at population=1000, rate=0.20
# ---------------------------------------------------------------------

def _make_config(strategy, rate):
    return Config(population_size=1000, contact_model="random-network",
                 random_degree_min=2, random_degree_max=10,
                 infection_probability=0.05, initial_infected=2,
                 simulation_days=10, random_seed=11,
                 vaccination_rate=rate, vaccination_strategy=strategy)


def test_random_vs_high_degree_same_count_same_network_seed():
    sim_random = Simulation(_make_config("random", 0.20))
    sim_hd = Simulation(_make_config("high_degree", 0.20))
    assert len(sim_random.vaccinated_ids) == len(sim_hd.vaccinated_ids) == 200

    graph = sim_random.engine.contact_model.graph
    degree = dict(graph.degree())
    avg_random = np.mean([degree[i] for i in sim_random.vaccinated_ids])
    avg_hd = np.mean([degree[i] for i in sim_hd.vaccinated_ids])
    assert avg_hd > avg_random


def test_zero_vaccination_identical_regardless_of_strategy():
    sim_random = Simulation(_make_config("random", 0.0))
    sim_hd = Simulation(_make_config("high_degree", 0.0))
    sim_random.run()
    sim_hd.run()
    assert sim_random.vaccinated_ids == sim_hd.vaccinated_ids == []
    assert ([r.__dict__ for r in sim_random.history]
            == [r.__dict__ for r in sim_hd.history])


def test_full_vaccination_prevents_all_transmission_both_strategies():
    for strategy in ("random", "high_degree"):
        config = _make_config(strategy, 1.0 - 2 / 1000)  # leave room for seeds
        sim = Simulation(config)
        sim.run()
        final = sim.history[-1]
        # Nearly everyone vaccinated -> epidemic can't spread beyond seeds.
        assert final.recovered + final.infectious + final.exposed <= 2


# ---------------------------------------------------------------------
# Part 3: population-aware initial infection policy
# ---------------------------------------------------------------------

@pytest.mark.parametrize("population", [100, 500, 1000, 10_000, 50_000, 100_000])
def test_population_aware_bounds(population):
    count = population_aware_initial_infected(population)
    assert 1 <= count <= 5


def test_population_aware_monotonic_nondecreasing():
    populations = [50, 100, 200, 500, 1000, 2000, 5000, 20_000, 60_000, 100_000]
    counts = [population_aware_initial_infected(p) for p in populations]
    assert all(b >= a for a, b in zip(counts, counts[1:]))


def test_population_aware_deterministic():
    assert (population_aware_initial_infected(12345)
            == population_aware_initial_infected(12345))


def test_fixed_policy_is_default_and_unchanged():
    config = Config(population_size=5000, initial_infected=2)
    assert config.initial_infection_policy == "fixed"
    assert config.resolved_initial_infected(5000) == 2


def test_population_aware_policy_is_opt_in():
    config = Config(population_size=100_000, initial_infected=2,
                    initial_infection_policy="population_aware")
    resolved = config.resolved_initial_infected(100_000)
    assert resolved != 2
    assert 1 <= resolved <= 5


# ---------------------------------------------------------------------
# Part 4/7: performance smoke test (no strict machine-specific thresholds)
# ---------------------------------------------------------------------

def test_small_simulation_completes_quickly():
    config = Config(population_size=500, contact_model="random-network",
                    simulation_days=30, random_seed=1)
    start = time.perf_counter()
    sim = Simulation(config)
    sim.run()
    elapsed = time.perf_counter() - start
    # Generous bound: this is a smoke test for gross regressions, not a
    # machine-calibrated performance assertion.
    assert elapsed < 30.0
