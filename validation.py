"""Validation suite: fast, self-checking invariants for the simulator.

Each check runs one or two small :class:`~regional_simulation.RegionalSimulation`
instances and asserts an invariant that must hold regardless of the specific
disease/travel parameters -- e.g. "zero infection probability never produces
new cases" or "the same seed reproduces the same run exactly". These are the
same invariants exercised by ``tests/test_validation.py`` (pytest); this
module additionally exposes a human-readable pass/fail report for
``python main.py --validate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from city import City, CityConfig
from config import Config
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


def check_travel_is_single_contact_not_network_broadcast() -> ValidationResult:
    """A visiting infectious traveler exposes at most one resident (the
    random host), never that host's whole neighbourhood/cluster.

    Regression guard: travel used to attach a visitor to a host's *entire*
    persistent contact list (their neighbours in the graph), so one
    infectious traveler could expose many residents in a single day -- a
    second, network-wide transmission channel layered on top of the city's
    own daily contacts. That made a city's internal dynamics diverge from an
    identical standalone (non-regional) run purely because it had active
    travelers. Travel must now be exactly one random inter-city link per
    traveler per day, independent of the destination's network.
    """
    from city import DiseaseToken
    from disease_model import State

    city_config = CityConfig(
        population_size=60, daily_contacts=6, infection_probability=1.0,
        incubation_days=2, infectious_days=5, contact_model_type="clustered",
        watts_strogatz_k=8, watts_strogatz_p=0.1, random_degree_min=1,
        random_degree_max=7, num_clusters=6, random_chance=0.0)
    city = City(0, city_config, np.random.default_rng(21))
    rng = np.random.default_rng(99)
    max_infected_per_visit = 0
    for day in range(200):
        # Fresh INFECTIOUS token every day (not aged/reused) so every call
        # actually exercises the infectious-visitor branch.
        token = DiseaseToken(state=State.INFECTIOUS, days_in_state=0)
        result = city.host_visitor_day(token, rng, day, "visitor")
        max_infected_per_visit = max(max_infected_per_visit,
                                     len(result.infected_resident_ids))
    passed = max_infected_per_visit <= 1
    return ValidationResult(
        "travel is a single inter-city contact, not a network broadcast",
        passed, f"max residents exposed by one visitor in one day = "
                f"{max_infected_per_visit} (must be <= 1)")


def check_clustered_contact_model_runs() -> ValidationResult:
    """The clustered contact model builds and runs for single and regional
    configs, including a per-city override alongside another model."""
    try:
        single = Config(number_of_cities=1, population_per_city=50,
                        contact_model="clustered", num_clusters=5,
                        random_chance=0.1, simulation_days=30, random_seed=7)
        _run(single)
        mixed = Config(city_populations=(50, 50),
                       clustered_cities=(0,),
                       num_clusters=5, random_chance=0.1,
                       simulation_days=30, random_seed=7)
        mixed_sim = _run(mixed)
        model_types = [
            type(city.engine.contact_model).__name__ for city in mixed_sim.cities]
        if model_types != ["ClusteredContactModel", "RandomNetworkContactModel"]:
            raise AssertionError(
                f"expected city 0 clustered / city 1 regular, got {model_types}")
        return ValidationResult("clustered contact model runs", True, "completed")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult("clustered contact model runs", False, repr(exc))


def check_clustered_cities_arbitrary_count_and_sizes() -> ValidationResult:
    """clustered_cities works for >2 cities with unequal populations, and
    each clustered city's cluster sizes scale to its own population."""
    try:
        config = Config(city_populations=(50, 200, 75, 500, 150),
                        clustered_cities=(0, 2, 4),
                        num_clusters=5, random_chance=0.1,
                        simulation_days=20, random_seed=13)
        sim = _run(config)
        expected_types = ["ClusteredContactModel", "RandomNetworkContactModel",
                          "ClusteredContactModel", "RandomNetworkContactModel",
                          "ClusteredContactModel"]
        actual_types = [type(c.engine.contact_model).__name__ for c in sim.cities]
        if actual_types != expected_types:
            raise AssertionError(f"expected {expected_types}, got {actual_types}")
        for city_id in (0, 2, 4):
            model = sim.cities[city_id].engine.contact_model
            sizes = sorted(len(c) for c in model.clusters)
            if max(sizes) - min(sizes) > 1:
                raise AssertionError(
                    f"city {city_id} cluster sizes too uneven: {sizes}")
        return ValidationResult(
            "clustered_cities: arbitrary city count and unequal populations",
            True, f"contact models = {actual_types}")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(
            "clustered_cities: arbitrary city count and unequal populations",
            False, repr(exc))


def check_clustered_cities_invalid_index_rejected() -> ValidationResult:
    """An out-of-range clustered_cities index raises a clear ValueError."""
    try:
        Config(city_populations=(50, 50), clustered_cities=(5,))
        return ValidationResult(
            "clustered_cities: invalid index is rejected", False,
            "expected ValueError, none raised")
    except ValueError as exc:
        return ValidationResult(
            "clustered_cities: invalid index is rejected", True, str(exc))


def check_clustered_zero_random_chance_is_segregated() -> ValidationResult:
    """With random_chance=0, no edge connects two different clusters."""
    city_config = CityConfig(
        population_size=60, daily_contacts=6, infection_probability=0.1,
        incubation_days=2, infectious_days=5, contact_model_type="clustered",
        watts_strogatz_k=8, watts_strogatz_p=0.1, random_degree_min=1,
        random_degree_max=7, num_clusters=6, random_chance=0.0)
    city = City(0, city_config, np.random.default_rng(8))
    model = city.engine.contact_model
    cross_edges = sum(1 for u, v in city.network.edges()
                      if model.cluster_of[u] != model.cluster_of[v])
    passed = cross_edges == 0
    return ValidationResult(
        "clustered model: random_chance=0 has no cross-cluster edges",
        passed, f"cross_cluster_edges={cross_edges}")


def check_clustered_reproducible() -> ValidationResult:
    """The same seed reproduces the same cluster assignment and edges."""
    city_config = CityConfig(
        population_size=60, daily_contacts=6, infection_probability=0.1,
        incubation_days=2, infectious_days=5, contact_model_type="clustered",
        watts_strogatz_k=8, watts_strogatz_p=0.1, random_degree_min=1,
        random_degree_max=7, num_clusters=6, random_chance=0.2)
    city_a = City(0, city_config, np.random.default_rng(11))
    city_b = City(0, city_config, np.random.default_rng(11))
    model_a, model_b = city_a.engine.contact_model, city_b.engine.contact_model
    passed = (list(model_a.cluster_of) == list(model_b.cluster_of)
             and sorted(city_a.network.edges()) == sorted(city_b.network.edges()))
    return ValidationResult(
        "clustered model is reproducible for a given seed",
        passed, "cluster assignment and edges matched" if passed
        else "cluster assignment or edges diverged")


def run_all_validations() -> List[ValidationResult]:
    """Run every validation check and return the results in a fixed order."""
    return [
        check_zero_infection_probability(),
        check_zero_travel(),
        check_travel_is_single_contact_not_network_broadcast(),
        check_same_seed_reproducible(),
        check_different_seed_varies(),
        check_small_population(),
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
