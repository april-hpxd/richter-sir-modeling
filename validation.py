"""Validation suite: fast, self-checking invariants for the simulator.

Each check runs one or two small :class:`~regional_simulation.RegionalSimulation`
instances and asserts an invariant that must hold regardless of the specific
disease/travel parameters -- e.g. "zero infection probability never produces
new cases" or "the same seed reproduces the same run exactly". 
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from city import City, CityConfig, DiseaseToken
from config import Config
from disease_model import State
from interaction import ClusteredContactModel
from regional_simulation import RegionalSimulation


@dataclass
class ValidationResult:
    """The outcome of one validation check."""

    name: str
    passed: bool
    detail: str


def _run(config: Config) -> RegionalSimulation:
    sim = RegionalSimulation(config)
    sim.run(verbose=False)
    return sim


def check_zero_infection_probability() -> ValidationResult:
    """With infection_probability=0, only the seeded cases ever get infected."""
    config = Config(number_of_cities=2, population_per_city=40,
                    infection_probability=0.0, initial_infected=2,
                    simulation_days=30, random_seed=1)
    sim = _run(config)
    summary = sim.regional_summary()
    passed = int(summary["total_infected"]) == config.initial_infected
    return ValidationResult(
        "zero infection probability",
        passed,
        f"total_infected={int(summary['total_infected'])} "
        f"(expected {config.initial_infected})")


def check_zero_travel() -> ValidationResult:
    """With travel_fraction=0, only the seed city (city 0) is ever infected."""
    config = Config(number_of_cities=3, population_per_city=60,
                    travel_fraction=0.0, daily_travel_rate=0.0,
                    initial_infected=2, infection_probability=0.2,
                    simulation_days=60, random_seed=2)
    sim = _run(config)
    summary = sim.regional_summary()
    passed = (summary["cities_reached"] == 1
             and int(summary["imported_infections"]) == 0)
    return ValidationResult(
        "zero travel",
        passed,
        f"cities_reached={summary['cities_reached']}/3, "
        f"imported_infections={int(summary['imported_infections'])}")


def check_travel_is_single_contact_not_network_broadcast() -> ValidationResult:
    """A visiting infectious person can expose at most one host per day."""
    city = City(1, CityConfig(
        population_size=12, daily_contacts=4, infection_probability=1.0,
        incubation_days=2, infectious_days=5, contact_model_type="well-mixed",
        watts_strogatz_k=2, watts_strogatz_p=0.1,
        random_degree_min=1, random_degree_max=7,
    ), np.random.default_rng(7))
    token = DiseaseToken(state=State.INFECTIOUS, days_in_state=0)
    result = city.host_visitor_day(token, np.random.default_rng(8), 1, "0-0")
    passed = len(result.infected_resident_ids) == 1
    return ValidationResult(
        "travel is a single contact",
        passed,
        f"infected_residents={len(result.infected_resident_ids)} (expected 1)")


def check_mixed_city_travel_transmission() -> ValidationResult:
    """A deterministic infectious traveler can seed a susceptible city."""
    config = Config(
        number_of_cities=2, population_per_city=8,
        infection_probability=1.0, initial_infected=1,
        simulation_days=3, travel_fraction=0.5, daily_travel_rate=0.0,
        random_seed=12,
    )
    sim = RegionalSimulation(config)
    for city in sim.cities:
        for individual in city.engine.individuals:
            individual.state = State.SUSCEPTIBLE
            individual.days_in_state = 0
            individual.present = True
    source = sim.cities[0].engine.individuals[0]
    source.state = State.INFECTIOUS
    sim.travel.eligible[0] = np.array([0], dtype=np.int64)
    sim.travel.matrix[:] = 0.0
    sim.travel.matrix[0, 1] = 1.0
    sim.run(verbose=False)
    imported = sim.regional_summary()["imported_infections"]
    passed = imported >= 1
    return ValidationResult(
        "mixed-city travel transmission", passed,
        f"imported_infections={int(imported)} (expected at least 1)")


def check_small_population_travel_eligibility() -> ValidationResult:
    """Small cities build a valid, bounded eligible traveler pool."""
    config = Config(number_of_cities=2, population_per_city=2,
                    travel_fraction=0.5, daily_travel_rate=0.5,
                    simulation_days=5, random_seed=13)
    sim = RegionalSimulation(config)
    sizes = [len(pool) for pool in sim.travel.eligible]
    passed = all(0 <= size <= 2 for size in sizes)
    return ValidationResult(
        "small-population travel eligibility", passed,
        f"eligible_pool_sizes={sizes}")


def check_travel_statistics_accounting() -> ValidationResult:
    """Travel summary counts agree with the recorded departure stream."""
    config = Config(number_of_cities=2, population_per_city=20,
                    travel_fraction=0.5, daily_travel_rate=0.2,
                    infection_probability=0.0, simulation_days=20,
                    random_seed=14)
    sim = _run(config)
    summary = sim.regional_summary()
    mobility = summary["mobility_statistics"]
    departures = int(summary["num_travel_events"])
    passed = (departures == sim.travel.total_departures
              and mobility["completed_trips"] <= departures
              and mobility["person_days_away"] >= 0)
    return ValidationResult(
        "travel statistics accounting", passed,
        f"departures={departures}, completed={mobility['completed_trips']}, "
        f"person_days_away={mobility['person_days_away']}")


def check_clustered_contact_model_runs() -> ValidationResult:
    """A regional simulation using clustered contacts completes."""
    try:
        config = Config(number_of_cities=2, population_per_city=30,
                        contact_model="clustered", num_clusters=5,
                        within_cluster_contact_probability=0.9,
                        simulation_days=15, random_seed=15)
        _run(config)
        return ValidationResult("clustered contact model runs", True, "completed")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult("clustered contact model runs", False, repr(exc))


def check_clustered_cities_arbitrary_count_and_sizes() -> ValidationResult:
    """Selected clustered cities support heterogeneous regional sizes."""
    try:
        config = Config(
            city_populations=(7, 11, 5), contact_model="daily-random",
            clustered_cities=(0, 2), num_clusters=3, simulation_days=10,
            random_seed=16,
        )
        _run(config)
        return ValidationResult(
            "clustered cities arbitrary count and sizes", True, "completed")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(
            "clustered cities arbitrary count and sizes", False, repr(exc))


def check_clustered_cities_invalid_index_rejected() -> ValidationResult:
    """Clustered-city indices outside the resolved city list are rejected."""
    try:
        Config(city_populations=(10, 10), clustered_cities=(2,))
    except ValueError as exc:
        return ValidationResult(
            "invalid clustered city index rejected", True, str(exc))
    return ValidationResult(
        "invalid clustered city index rejected", False,
        "invalid index was accepted")


def check_clustered_zero_random_chance_is_segregated() -> ValidationResult:
    """With no between-cluster chance, every sampled contact stays local."""
    model = ClusteredContactModel(
        population_size=20, num_clusters=4,
        within_cluster_contact_probability=1.0,
        min_degree=1, max_degree=3, rng=np.random.default_rng(17),
    )
    local = all(
        int(model.cluster_of[person_id]) == int(model.cluster_of[partner])
        for person_id, contacts in enumerate(model.contact_lists)
        for partner in contacts
    )
    return ValidationResult(
        "zero between-cluster chance is segregated", local,
        "all contacts stayed within their assigned cluster"
        if local else "a cross-cluster contact was generated")


def check_clustered_reproducible() -> ValidationResult:
    """Identical clustered configurations reproduce exactly."""
    config = Config(population_size=40, contact_model="clustered",
                    num_clusters=5, within_cluster_contact_probability=0.9,
                    simulation_days=20, random_seed=18)
    first = _run(config)
    second = _run(config)
    passed = first.history == second.history
    return ValidationResult(
        "clustered simulation is reproducible", passed,
        "history matched exactly" if passed else "history diverged")


def check_same_seed_reproducible() -> ValidationResult:
    """Two runs with an identical Config (same seed) produce identical history."""
    config = Config(number_of_cities=2, population_per_city=50,
                    travel_fraction=0.3, daily_travel_rate=0.1,
                    random_seed=3, simulation_days=40)
    sim_a = _run(config)
    sim_b = _run(config)
    passed = sim_a.history == sim_b.history
    return ValidationResult(
        "same seed is reproducible",
        passed,
        "history matched exactly" if passed else "history diverged")


def check_different_seed_varies() -> ValidationResult:
    """Two runs differing only by seed produce a different outcome."""
    base = Config(number_of_cities=2, population_per_city=50,
                 travel_fraction=0.3, daily_travel_rate=0.1,
                 simulation_days=40)
    sim_a = _run(base.with_overrides(random_seed=10))
    sim_b = _run(base.with_overrides(random_seed=11))
    passed = sim_a.history != sim_b.history
    return ValidationResult(
        "different seed varies the outcome",
        passed,
        "history differed as expected" if passed
        else "history was identical across different seeds")


def check_small_population() -> ValidationResult:
    """A minimal 2-person city runs to completion without error."""
    try:
        config = Config(number_of_cities=1, population_per_city=2,
                        initial_infected=1, daily_contacts=1,
                        contact_model="well-mixed", simulation_days=30,
                        random_seed=4)
        _run(config)
        return ValidationResult("small population (n=2) runs", True, "completed")
    except Exception as exc:  # noqa: BLE001 -- report any failure as a result
        return ValidationResult("small population (n=2) runs", False, repr(exc))


def check_large_population() -> ValidationResult:
    """A 2000-person city runs to completion without error."""
    try:
        config = Config(number_of_cities=1, population_per_city=2000,
                        initial_infected=5, simulation_days=60,
                        random_seed=5)
        _run(config)
        return ValidationResult("large population (n=2000) runs", True, "completed")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult("large population (n=2000) runs", False, repr(exc))


def check_city_counts() -> ValidationResult:
    """City counts 1, 2, 5, and 10 all run to completion without error."""
    for n in (1, 2, 5, 10):
        try:
            config = Config(number_of_cities=n, population_per_city=30,
                            simulation_days=30, random_seed=6)
            _run(config)
        except Exception as exc:  # noqa: BLE001
            return ValidationResult(
                "city counts (1, 2, 5, 10) run", False,
                f"failed at number_of_cities={n}: {exc!r}")
    return ValidationResult("city counts (1, 2, 5, 10) run", True, "completed")


def run_all_validations() -> List[ValidationResult]:
    """Run every validation check and return the results in a fixed order."""
    return [
        check_zero_infection_probability(),
        check_zero_travel(),
        check_travel_is_single_contact_not_network_broadcast(),
        check_mixed_city_travel_transmission(),
        check_same_seed_reproducible(),
        check_different_seed_varies(),
        check_small_population(),
        check_small_population_travel_eligibility(),
        check_travel_statistics_accounting(),
        check_large_population(),
        check_city_counts(),
        check_clustered_contact_model_runs(),
        check_clustered_cities_arbitrary_count_and_sizes(),
        check_clustered_cities_invalid_index_rejected(),
        check_clustered_zero_random_chance_is_segregated(),
        check_clustered_reproducible(),
    ]


def print_validation_report(results: List[ValidationResult]) -> None:
    """Print a pass/fail report for a list of :class:`ValidationResult`."""
    print("\n" + "=" * 60)
    print("  VALIDATION SUITE")
    print("=" * 60)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.name}")
        print(f"         {result.detail}")
    passed = sum(1 for r in results if r.passed)
    print("-" * 60)
    print(f"  {passed}/{len(results)} checks passed")
    print("=" * 60 + "\n")
