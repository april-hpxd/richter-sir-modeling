from city import DiseaseToken
from config import Config
from disease_model import State
from regional_simulation import RegionalSimulation
from travel import Traveler
from validation import (
    check_city_counts,
    check_clustered_cities_arbitrary_count_and_sizes,
    check_clustered_cities_invalid_index_rejected,
    check_clustered_contact_model_runs,
    check_clustered_reproducible,
    check_clustered_zero_random_chance_is_segregated,
    check_different_seed_varies,
    check_large_population,
    check_same_seed_reproducible,
    check_small_population,
    check_zero_infection_probability,
    check_zero_travel,
)


def test_zero_infection_probability():
    result = check_zero_infection_probability()
    assert result.passed, result.detail


def test_zero_travel():
    result = check_zero_travel()
    assert result.passed, result.detail


def test_same_seed_reproducible():
    result = check_same_seed_reproducible()
    assert result.passed, result.detail


def test_different_seed_varies():
    result = check_different_seed_varies()
    assert result.passed, result.detail


def test_small_population():
    result = check_small_population()
    assert result.passed, result.detail


def test_large_population():
    result = check_large_population()
    assert result.passed, result.detail


def test_city_counts():
    result = check_city_counts()
    assert result.passed, result.detail


def test_clustered_contact_model_runs():
    result = check_clustered_contact_model_runs()
    assert result.passed, result.detail


def test_clustered_cities_arbitrary_count_and_sizes():
    result = check_clustered_cities_arbitrary_count_and_sizes()
    assert result.passed, result.detail


def test_clustered_cities_invalid_index_rejected():
    result = check_clustered_cities_invalid_index_rejected()
    assert result.passed, result.detail


def test_clustered_zero_random_chance_is_segregated():
    result = check_clustered_zero_random_chance_is_segregated()
    assert result.passed, result.detail


def test_clustered_reproducible():
    result = check_clustered_reproducible()
    assert result.passed, result.detail


def test_regional_run_keeps_going_for_active_travelers():
    config = Config(
        number_of_cities=2,
        population_per_city=10,
        simulation_days=5,
        infection_probability=0.0,
        initial_infected=1,
        random_seed=9,
        travel_fraction=0.2,
        daily_travel_rate=0.5,
    )
    sim = RegionalSimulation(config)

    for city in sim.cities:
        for ind in city.engine.individuals:
            ind.state = State.RECOVERED
            ind.days_in_state = 0
            ind.present = True
        city.engine.individuals[0].state = State.SUSCEPTIBLE

    sim.travel.active = [
        Traveler(
            home_city_id=0,
            home_individual_id=0,
            current_city_id=1,
            days_remaining=2,
            token=DiseaseToken(state=State.INFECTIOUS, days_in_state=0),
            state_before=State.INFECTIOUS,
            day_departed=0,
        )
    ]
    sim.travel._away[0].add(0)

    sim.run(verbose=False)

    assert len(sim.history) > 2
    assert len(sim.travel.travel_events) >= 1
