from config import Config
from epidemic_stats import is_duration_censored, summary
from regional_simulation import RegionalSimulation
from simulation import Simulation, DailyRecord


def test_epidemic_that_dies_out_is_not_censored():
    config = Config(population_size=60, contact_model="well-mixed",
                    daily_contacts=6, infection_probability=0.3,
                    infectious_days=3, initial_infected=2,
                    simulation_days=200, random_seed=11)
    sim = Simulation(config)
    history = sim.run()
    # The run must have stopped early (extinction), not hit the horizon.
    assert history[-1].day < config.simulation_days
    assert is_duration_censored(history) is False
    assert summary(history)["duration_censored"] is False


def test_epidemic_still_active_at_horizon_is_censored():
    config = Config(population_size=300, contact_model="well-mixed",
                    daily_contacts=10, infection_probability=0.5,
                    infectious_days=30, initial_infected=5,
                    simulation_days=3, random_seed=5)
    sim = Simulation(config)
    history = sim.run()
    assert history[-1].day == config.simulation_days
    assert is_duration_censored(history) is True
    assert summary(history)["duration_censored"] is True


def test_is_duration_censored_empty_history_is_false():
    assert is_duration_censored([]) is False


def test_is_duration_censored_manual_records():
    still_active = [DailyRecord(day=0, susceptible=8, exposed=1, infectious=1,
                                recovered=0, new_exposed=1, new_infectious=0,
                                new_recovered=0)]
    extinct = [DailyRecord(day=0, susceptible=8, exposed=0, infectious=0,
                           recovered=2, new_exposed=0, new_infectious=0,
                           new_recovered=0)]
    assert is_duration_censored(still_active) is True
    assert is_duration_censored(extinct) is False


def test_city_summary_stats_duration_censored_flag():
    import numpy as np
    from city import City, CityConfig

    city_config = CityConfig(
        population_size=200, daily_contacts=10, infection_probability=0.5,
        incubation_days=1, infectious_days=30,
        contact_model_type="well-mixed", watts_strogatz_k=8,
        watts_strogatz_p=0.1, random_degree_min=1, random_degree_max=7,
    )
    city = City(city_id=0, config=city_config, rng=np.random.default_rng(5))
    city.seed_infection(5)
    city.run(simulation_days=3)
    stats = city.summary_stats()
    assert stats["duration_censored"] is True

    # A fast-burning, quickly-extinguished outbreak should not be censored.
    city_config2 = CityConfig(
        population_size=60, daily_contacts=6, infection_probability=0.3,
        incubation_days=1, infectious_days=3,
        contact_model_type="well-mixed", watts_strogatz_k=8,
        watts_strogatz_p=0.1, random_degree_min=1, random_degree_max=7,
    )
    city2 = City(city_id=0, config=city_config2, rng=np.random.default_rng(11))
    city2.seed_infection(2)
    city2.run(simulation_days=200)
    stats2 = city2.summary_stats()
    assert stats2["duration_censored"] is False


def test_regional_summary_duration_censored_fields():
    config = Config(city_populations=(80, 80), contact_model="well-mixed",
                    daily_contacts=10, infection_probability=0.6,
                    infectious_days=30, initial_infected=5,
                    daily_travel_rate=0.2, travel_fraction=0.3,
                    simulation_days=3, random_seed=9)
    regional = RegionalSimulation(config)
    regional.run()
    summary_dict = regional.regional_summary()
    assert "regional_duration_censored" in summary_dict
    assert "cities_duration_censored" in summary_dict
    assert len(summary_dict["cities_duration_censored"]) == 2
    assert summary_dict["regional_duration_censored"] is True
