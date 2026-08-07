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


def run_all_validations() -> List[ValidationResult]:
    """Run every validation check and return the results in a fixed order."""
    return [
        check_zero_infection_probability(),
        check_zero_travel(),
        check_same_seed_reproducible(),
        check_different_seed_varies(),
        check_small_population(),
        check_large_population(),
        check_city_counts(),
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
