"""Tests for the explicit termination-metadata fields added in the final
validation milestone: final_exposed, final_infectious, extinction_reached,
actual_days_run, and (at the RunResult/regional level) simulation_max_days,
termination_reason, and major_outbreak.

These fields are read directly off the true final day's state -- the same
state ``duration_censored``/``is_duration_censored`` already used -- so they
should always agree with it by construction, never be independently wrong.
"""

import numpy as np

from city import City, CityConfig
from config import Config
from epidemic_stats import summary
from experiments import DEFAULT_MAJOR_OUTBREAK_THRESHOLD, _summarise_run
from regional_simulation import RegionalSimulation
from simulation import DailyRecord, Simulation


def _city(population_size, infectious_days, incubation_days=1,
         daily_contacts=10, infection_probability=0.5, seed=5):
    config = CityConfig(
        population_size=population_size, daily_contacts=daily_contacts,
        infection_probability=infection_probability,
        incubation_days=incubation_days, infectious_days=infectious_days,
        contact_model_type="well-mixed", watts_strogatz_k=8,
        watts_strogatz_p=0.1, random_degree_min=1, random_degree_max=7,
    )
    return City(city_id=0, config=config, rng=np.random.default_rng(seed))


# ---------------------------------------------------------------------
# Single-city (epidemic_stats.summary) termination metadata
# ---------------------------------------------------------------------

def test_immediate_extinction_no_e_or_i_at_start():
    """No E/I to begin with: day 0 alone is already extinct."""
    history = [DailyRecord(day=0, susceptible=10, exposed=0, infectious=0,
                           recovered=0, new_exposed=0, new_infectious=0,
                           new_recovered=0)]
    stats = summary(history)
    assert stats["final_exposed"] == 0
    assert stats["final_infectious"] == 0
    assert stats["extinction_reached"] is True
    assert stats["duration_censored"] is False
    assert stats["actual_days_run"] == 0


def test_i_zero_but_e_positive_is_not_falsely_extinct():
    """I==0 with E>0 remaining must NOT be classified as extinct."""
    history = [DailyRecord(day=0, susceptible=8, exposed=2, infectious=0,
                           recovered=0, new_exposed=2, new_infectious=0,
                           new_recovered=0)]
    stats = summary(history)
    assert stats["final_exposed"] == 2
    assert stats["final_infectious"] == 0
    assert stats["extinction_reached"] is False
    assert stats["duration_censored"] is True


def test_extinction_before_max_days():
    config = Config(population_size=60, contact_model="well-mixed",
                    daily_contacts=6, infection_probability=0.3,
                    infectious_days=3, initial_infected=2,
                    simulation_days=200, random_seed=11)
    sim = Simulation(config)
    history = sim.run()
    stats = summary(history)
    assert history[-1].day < config.simulation_days
    assert stats["extinction_reached"] is True
    assert stats["final_exposed"] == 0
    assert stats["final_infectious"] == 0
    assert stats["duration_censored"] is False
    assert stats["actual_days_run"] == history[-1].day


def test_max_days_hit_with_infectious_remaining():
    config = Config(population_size=300, contact_model="well-mixed",
                    daily_contacts=10, infection_probability=0.5,
                    infectious_days=30, initial_infected=5,
                    simulation_days=3, random_seed=5)
    sim = Simulation(config)
    history = sim.run()
    stats = summary(history)
    assert history[-1].day == config.simulation_days
    assert stats["extinction_reached"] is False
    assert stats["final_infectious"] > 0 or stats["final_exposed"] > 0
    assert stats["duration_censored"] is True
    assert stats["actual_days_run"] == config.simulation_days


def test_max_days_hit_with_exposed_only_remaining():
    """Force a stop while only E (no I) remains: short horizon, long
    incubation, so everyone seeded is still incubating when the cap hits."""
    history = [
        DailyRecord(day=0, susceptible=5, exposed=5, infectious=0,
                   recovered=0, new_exposed=5, new_infectious=0,
                   new_recovered=0),
        DailyRecord(day=1, susceptible=5, exposed=5, infectious=0,
                   recovered=0, new_exposed=0, new_infectious=0,
                   new_recovered=0),
    ]
    stats = summary(history)
    assert stats["final_exposed"] == 5
    assert stats["final_infectious"] == 0
    assert stats["extinction_reached"] is False
    assert stats["duration_censored"] is True
    assert stats["actual_days_run"] == 1


def test_extinction_exactly_on_final_permitted_day():
    """If the last allowed day happens to be the extinction day, the final
    day's true state (not loop-iteration count) must drive classification."""
    history = [
        DailyRecord(day=0, susceptible=5, exposed=0, infectious=3,
                   recovered=2, new_exposed=0, new_infectious=0,
                   new_recovered=0),
        DailyRecord(day=1, susceptible=5, exposed=0, infectious=0,
                   recovered=5, new_exposed=0, new_infectious=0,
                   new_recovered=3),
    ]
    stats = summary(history)
    assert stats["actual_days_run"] == 1
    assert stats["extinction_reached"] is True
    assert stats["final_exposed"] == 0
    assert stats["final_infectious"] == 0
    assert stats["duration_censored"] is False


# ---------------------------------------------------------------------
# Single-city path via City.summary_stats()
# ---------------------------------------------------------------------

def test_city_summary_stats_termination_fields_extinction():
    city = _city(population_size=60, infectious_days=3,
                infection_probability=0.3, daily_contacts=6, seed=11)
    city.seed_infection(2)
    city.run(simulation_days=200)
    stats = city.summary_stats()
    assert stats["extinction_reached"] is True
    assert stats["final_exposed"] == 0
    assert stats["final_infectious"] == 0
    assert stats["duration_censored"] is False
    assert stats["actual_days_run"] < 200


def test_city_summary_stats_termination_fields_censored():
    city = _city(population_size=200, infectious_days=30,
                infection_probability=0.5, daily_contacts=10, seed=5)
    city.seed_infection(5)
    city.run(simulation_days=3)
    stats = city.summary_stats()
    assert stats["extinction_reached"] is False
    assert stats["final_exposed"] > 0 or stats["final_infectious"] > 0
    assert stats["duration_censored"] is True
    assert stats["actual_days_run"] == 3


def test_city_summary_stats_termination_fields_empty_history():
    city = _city(population_size=60, infectious_days=3, seed=1)
    stats = city.summary_stats()
    assert stats["extinction_reached"] is True
    assert stats["final_exposed"] == 0
    assert stats["final_infectious"] == 0
    assert stats["actual_days_run"] == 0


# ---------------------------------------------------------------------
# Regional path via RunResult / _summarise_run
# ---------------------------------------------------------------------

def test_regional_run_result_termination_fields_extinction():
    config = Config(city_populations=(60, 60), contact_model="daily-random",
                    daily_contacts=6, infection_probability=0.3,
                    infectious_days=3, initial_infected=2,
                    daily_travel_rate=0.1, travel_fraction=0.1,
                    num_clusters=2,
                    simulation_days=200, random_seed=11)
    sim = RegionalSimulation(config)
    sim.run()
    result = _summarise_run(sim, seed=11)
    assert result.extinction_reached is True
    assert result.final_exposed == 0
    assert result.final_infectious == 0
    assert result.duration_censored is False
    assert result.termination_reason == "extinction"
    assert result.simulation_max_days == 200
    assert result.actual_days_run < 200


def test_regional_run_result_termination_fields_censored():
    config = Config(city_populations=(80, 80), contact_model="daily-random",
                    daily_contacts=10, infection_probability=0.6,
                    infectious_days=30, initial_infected=5,
                    daily_travel_rate=0.2, travel_fraction=0.3,
                    num_clusters=2,
                    simulation_days=3, random_seed=9)
    sim = RegionalSimulation(config)
    sim.run()
    result = _summarise_run(sim, seed=9)
    assert result.duration_censored is True
    assert result.extinction_reached is False
    assert result.final_exposed + result.final_infectious > 0
    assert result.termination_reason == "max_days"
    assert result.simulation_max_days == 3
    assert result.actual_days_run == 3


def test_regional_single_city_and_single_city_agree_on_extinction_logic():
    """Same underlying rule (final E==0 and I==0) drives both paths."""
    single_config = Config(population_size=60, contact_model="well-mixed",
                           daily_contacts=6, infection_probability=0.3,
                           infectious_days=3, initial_infected=2,
                           simulation_days=200, random_seed=11)
    single = Simulation(single_config)
    single_stats = summary(single.run())

    regional_config = Config(city_populations=(60,), contact_model="daily-random",
                             daily_contacts=6, infection_probability=0.3,
                             infectious_days=3, initial_infected=2,
                             num_clusters=2,
                             simulation_days=200, random_seed=11)
    regional = RegionalSimulation(regional_config)
    regional.run()
    regional_result = _summarise_run(regional, seed=11)

    assert single_stats["extinction_reached"] == regional_result.extinction_reached
    assert single_stats["duration_censored"] == regional_result.duration_censored


# ---------------------------------------------------------------------
# major_outbreak classification (boundary tests)
# ---------------------------------------------------------------------

def test_major_outbreak_threshold_boundaries():
    from experiments import is_major_outbreak, RunResult

    def make_result(attack_rate):
        return RunResult(
            seed=0, scenario="well-mixed", experiment_name="", replicate=0,
            population=1000, initial_infected=5, contact_model="well-mixed",
            infection_probability=0.3, incubation_days=2, infectious_days=6,
            simulation_days=200, daily_travel_rate=0.0, travel_fraction=0.0,
            cluster_count=1, within_cluster_probability=1.0,
            average_contacts=10.0, std_contacts=0.0,
            vaccination_rate=0.0, vaccination_strategy="random",
            network_node_count=1000.0, network_edge_count=5000.0,
            network_mean_degree=10.0, network_std_degree=0.0,
            network_clustering_coefficient=0.0,
            network_num_connected_components=1.0,
            network_largest_component_size=1000.0,
            network_average_shortest_path_length=0.0,
            network_path_length_computed=False,
            average_arrival_delay=-1.0, cities_reached=1,
            peak_regional_infectious=0, peak_regional_infectious_day=0,
            epidemic_duration=10, duration_censored=False,
            total_infected=int(attack_rate * 1000), attack_rate=attack_rate,
            imported_infections=0, first_infection_days=[0.0],
            final_exposed=0, final_infectious=0, extinction_reached=True,
            actual_days_run=10, simulation_max_days=200,
            termination_reason="extinction",
            major_outbreak=(attack_rate >= DEFAULT_MAJOR_OUTBREAK_THRESHOLD),
        )

    just_below = make_result(0.199)
    exactly_at = make_result(0.20)
    just_above = make_result(0.201)

    assert is_major_outbreak(just_below) is False
    assert is_major_outbreak(exactly_at) is True
    assert is_major_outbreak(just_above) is True
    assert just_below.major_outbreak is False
    assert exactly_at.major_outbreak is True
    assert just_above.major_outbreak is True
