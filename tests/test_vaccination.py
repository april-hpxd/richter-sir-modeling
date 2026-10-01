import numpy as np
import pytest

from city import City, CityConfig
from config import Config
from disease_model import State
from simulation import Simulation
from vaccination import vaccinate_random


def _engine_individuals(population_size):
    """Build a bare population (all SUSCEPTIBLE) for unit-testing vaccinate_random."""
    from disease_model import Individual
    return [Individual(id=i) for i in range(population_size)]


# 1. No vaccination behaves as before.
def test_no_vaccination_matches_pre_vaccination_behaviour():
    config_a = Config(population_size=60, contact_model="well-mixed",
                      daily_contacts=6, infection_probability=0.2,
                      initial_infected=3, simulation_days=40, random_seed=7)
    config_b = Config(population_size=60, contact_model="well-mixed",
                      daily_contacts=6, infection_probability=0.2,
                      initial_infected=3, simulation_days=40, random_seed=7,
                      vaccination_rate=0.0)
    sim_a = Simulation(config_a)
    sim_a.run()
    sim_b = Simulation(config_b)
    sim_b.run()
    # Identical RNG draws end to end: the vaccination step must not consume
    # any randomness when disabled.
    assert [r.__dict__ for r in sim_a.history] == [r.__dict__ for r in sim_b.history]
    assert sim_b.vaccinated_ids == []
    assert sim_b.vaccination_report()["vaccination_enabled"] is False


# 2. Requested vaccination count is correct.
def test_vaccinate_random_selects_exact_count():
    individuals = _engine_individuals(100)
    chosen = vaccinate_random(individuals, 25, np.random.default_rng(1))
    assert len(chosen) == 25
    assert sum(1 for ind in individuals if ind.state is State.VACCINATED) == 25


# 3. No node is vaccinated twice.
def test_vaccinate_random_never_duplicates_a_node():
    individuals = _engine_individuals(50)
    chosen = vaccinate_random(individuals, 30, np.random.default_rng(2))
    assert len(chosen) == len(set(chosen))


# 4. Vaccinated nodes cannot become infected.
def test_vaccinated_nodes_never_become_infected():
    config = Config(population_size=80, contact_model="well-mixed",
                    daily_contacts=10, infection_probability=1.0,
                    initial_infected=3, simulation_days=60, random_seed=3,
                    vaccination_rate=0.5)
    sim = Simulation(config)
    sim.run()
    vaccinated_ids = set(sim.vaccinated_ids)
    assert vaccinated_ids
    for ind in sim.engine.individuals:
        if ind.id in vaccinated_ids:
            assert ind.state is State.VACCINATED


# 5. Vaccinated nodes cannot transmit infection.
def test_vaccinated_nodes_never_become_infectious_and_cannot_transmit():
    config = Config(population_size=80, contact_model="well-mixed",
                    daily_contacts=10, infection_probability=1.0,
                    initial_infected=3, simulation_days=60, random_seed=4,
                    vaccination_rate=0.5)
    sim = Simulation(config)
    for record in sim.run():
        pass
    vaccinated_ids = set(sim.vaccinated_ids)
    # No transmission in the recorded history was ever attributed to a
    # vaccinated source id.
    for frame in sim.transmission_frames:
        for source_id, _target_id in frame:
            assert source_id not in vaccinated_ids
    for ind in sim.engine.individuals:
        assert not (ind.id in vaccinated_ids and ind.state is State.INFECTIOUS)


# 6. Vaccinated nodes remain vaccinated for the whole run.
def test_vaccinated_nodes_remain_vaccinated_the_whole_run():
    config = Config(population_size=60, contact_model="well-mixed",
                    daily_contacts=8, infection_probability=0.4,
                    initial_infected=2, simulation_days=50, random_seed=5,
                    vaccination_rate=0.4)
    sim = Simulation(config)
    sim.run()
    vaccinated_ids = set(sim.vaccinated_ids)
    for frame in sim.state_frames:
        for vid in vaccinated_ids:
            assert frame[vid] is State.VACCINATED


# 7. Cannot vaccinate more people than the population.
def test_cannot_vaccinate_more_than_population():
    individuals = _engine_individuals(10)
    with pytest.raises(ValueError):
        vaccinate_random(individuals, 11, np.random.default_rng(6))


def test_vaccination_rate_above_one_is_rejected_by_config():
    with pytest.raises(ValueError):
        Config(vaccination_rate=1.5)


# 8. Same seed produces the same vaccinated individuals.
def test_same_seed_reproduces_same_vaccinated_set():
    config = Config(population_size=100, random_seed=42, vaccination_rate=0.2,
                    initial_infected=2)
    sim_a = Simulation(config)
    sim_b = Simulation(config)
    assert sim_a.vaccinated_ids == sim_b.vaccinated_ids
    assert len(sim_a.vaccinated_ids) == 20


# 9. Different seeds can produce different vaccination selections.
def test_different_seeds_can_select_different_vaccinated_sets():
    config_a = Config(population_size=100, random_seed=1, vaccination_rate=0.2,
                      initial_infected=2)
    config_b = Config(population_size=100, random_seed=2, vaccination_rate=0.2,
                      initial_infected=2)
    sim_a = Simulation(config_a)
    sim_b = Simulation(config_b)
    assert sim_a.vaccinated_ids != sim_b.vaccinated_ids


# 10. Small populations work correctly.
def test_small_population_vaccination():
    individuals = _engine_individuals(2)
    chosen = vaccinate_random(individuals, 1, np.random.default_rng(9))
    assert len(chosen) == 1
    config = Config(population_size=2, initial_infected=1, random_seed=1,
                    vaccination_rate=0.5, daily_contacts=1, num_clusters=1)
    sim = Simulation(config)
    assert len(sim.vaccinated_ids) == 1


# 11. Zero vaccination works correctly.
def test_zero_vaccination_count_is_a_no_op():
    individuals = _engine_individuals(20)
    chosen = vaccinate_random(individuals, 0, np.random.default_rng(10))
    assert chosen == []
    assert all(ind.state is State.SUSCEPTIBLE for ind in individuals)

    config = Config(population_size=40, random_seed=1, vaccination_rate=0.0,
                    initial_infected=2)
    sim = Simulation(config)
    assert sim.vaccinated_ids == []


# 12. Full vaccination prevents infection when there is no pre-existing
# infected individual outside the vaccinated set. Initial cases are seeded
# (as EXPOSED) from whoever is still SUSCEPTIBLE after vaccination runs, so
# vaccinating exactly `population_size - initial_infected` people leaves
# precisely the to-be-seeded individuals unvaccinated -- nobody susceptible
# remains once seeding happens, so the disease can never spread further.
def test_full_vaccination_of_everyone_else_prevents_further_spread():
    population_size = 50
    initial_infected = 2
    config = Config(population_size=population_size, contact_model="well-mixed",
                    daily_contacts=10, infection_probability=1.0,
                    incubation_days=1, infectious_days=3,
                    initial_infected=initial_infected, simulation_days=30,
                    random_seed=8,
                    vaccination_rate=(population_size - initial_infected) / population_size)
    sim = Simulation(config)
    assert len(sim.vaccinated_ids) == population_size - initial_infected
    history = sim.run()
    # Nobody beyond the originally-seeded cases was ever exposed: vaccinated
    # individuals are never susceptible, so the seeded cases have no
    # susceptible contacts left to infect.
    assert sum(r.new_exposed for r in history[1:]) == 0
    final = history[-1]
    assert final.vaccinated == population_size - initial_infected
    assert final.recovered + final.infectious + final.exposed == initial_infected


# Experiment compatibility -------------------------------------------------

def test_vaccination_works_in_city_single_city_standalone():
    city_config = CityConfig(
        population_size=60, daily_contacts=6, infection_probability=0.3,
        incubation_days=2, infectious_days=5, contact_model_type="well-mixed",
        watts_strogatz_k=8, watts_strogatz_p=0.1, random_degree_min=1,
        random_degree_max=7, vaccination_count=15,
    )
    city = City(city_id=0, config=city_config, rng=np.random.default_rng(3))
    assert len(city.vaccinated_ids) == 15
    city.seed_infection(3)
    city.run(simulation_days=30)
    vaccinated_ids = set(city.vaccinated_ids)
    for ind in city.engine.individuals:
        if ind.id in vaccinated_ids:
            assert ind.state is State.VACCINATED
    report = city.vaccination_report()
    assert report["vaccination_enabled"] is True
    assert report["number_vaccinated"] == 15


def test_vaccination_works_in_two_city_regional_run():
    from regional_simulation import RegionalSimulation

    config = Config(number_of_cities=2, population_per_city=60,
                    contact_model="daily-random", infection_probability=0.3,
                    initial_infected=2, travel_fraction=0.3,
                    daily_travel_rate=0.1, simulation_days=40,
                    random_seed=11, vaccination_rate=0.25)
    sim = RegionalSimulation(config)
    for city in sim.cities:
        assert len(city.vaccinated_ids) == round(0.25 * city.config.population_size)
    sim.run()
    summary = sim.regional_summary()
    assert summary["total_vaccinated"] == sum(
        len(c.vaccinated_ids) for c in sim.cities)
    assert len(summary["vaccination_reports"]) == 2
    for ind_city in sim.cities:
        vaccinated_ids = set(ind_city.vaccinated_ids)
        for ind in ind_city.engine.individuals:
            if ind.id in vaccinated_ids:
                assert ind.state is State.VACCINATED


@pytest.mark.parametrize("contact_model", [
    "well-mixed", "random-network", "watts-strogatz", "daily-random", "clustered",
])
def test_vaccination_works_across_existing_network_types(contact_model):
    config = Config(population_size=60, contact_model=contact_model,
                    daily_contacts=6, random_degree_min=2, random_degree_max=6,
                    watts_strogatz_k=4, num_clusters=4, infection_probability=0.2,
                    initial_infected=2, simulation_days=20, random_seed=13,
                    vaccination_rate=0.2)
    sim = Simulation(config)
    assert len(sim.vaccinated_ids) == round(0.2 * 60)
    sim.run()


def test_vaccination_works_with_experiment_runner():
    from experiments import run_experiment

    config = Config(number_of_cities=1, population_per_city=50,
                    infection_probability=0.2, initial_infected=2,
                    simulation_days=20, vaccination_rate=0.3)
    result = run_experiment(config, num_runs=2, base_seed=0, verbose=False)
    assert result["num_runs"] == 2
