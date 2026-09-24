import dataclasses
import os

from config import Config
from experiments import (
    RunResult,
    run_experiment,
    run_scenario_comparison,
    write_run_results_csv,
    write_scenario_comparison_csv,
)


def _base_config(**overrides):
    defaults = dict(
        city_populations=(30, 30), simulation_days=30,
        daily_travel_rate=0.1, travel_fraction=0.3,
        initial_infected=2, num_clusters=3, contact_model="daily-random",
    )
    defaults.update(overrides)
    return Config(**defaults)


def test_run_experiment_replicate_count_and_seeds_are_configurable():
    result = run_experiment(_base_config(), num_runs=4, base_seed=500)
    assert len(result["runs"]) == 4
    assert [r.seed for r in result["runs"]] == [500, 501, 502, 503]
    assert [r.replicate for r in result["runs"]] == [0, 1, 2, 3]


def test_run_experiment_is_reproducible_with_same_seed_and_config():
    config = _base_config()
    first = run_experiment(config, num_runs=3, base_seed=42)
    second = run_experiment(config, num_runs=3, base_seed=42)
    for r1, r2 in zip(first["runs"], second["runs"]):
        assert dataclasses.asdict(r1) == dataclasses.asdict(r2)


def test_run_experiment_different_seeds_can_diverge():
    config = _base_config(infection_probability=0.5)
    a = run_experiment(config, num_runs=1, base_seed=1)
    b = run_experiment(config, num_runs=1, base_seed=2)
    # Not a strict guarantee for every possible config, but true for this one
    # (different seeds should not be forced to collapse to the same outcome).
    fields_a = {k: v for k, v in dataclasses.asdict(a["runs"][0]).items()
               if k not in ("seed", "replicate")}
    fields_b = {k: v for k, v in dataclasses.asdict(b["runs"][0]).items()
               if k not in ("seed", "replicate")}
    assert fields_a != fields_b


def test_run_result_carries_complete_metadata():
    result = run_experiment(_base_config(), num_runs=1, base_seed=7)
    run = result["runs"][0]
    assert isinstance(run, RunResult)
    fields = {f.name for f in dataclasses.fields(RunResult)}
    # Identification, model, network, and outcome groups must all be present.
    for expected in (
        "seed", "scenario", "experiment_name", "replicate",
        "population", "initial_infected", "infection_probability",
        "simulation_days", "daily_travel_rate", "travel_fraction",
        "network_node_count", "network_mean_degree",
        "network_clustering_coefficient", "network_num_connected_components",
        "network_largest_component_size", "network_average_shortest_path_length",
        "average_arrival_delay", "cities_reached", "epidemic_duration",
        "duration_censored", "total_infected", "attack_rate",
        "imported_infections",
    ):
        assert expected in fields, expected


def test_run_scenario_comparison_runs_named_scenarios_with_shared_seeds():
    scenarios = {
        "clustered": _base_config(clustered_cities=(0,), random_chance=0.1),
        "random": _base_config(),
    }
    comparison = run_scenario_comparison(scenarios, num_runs=3, base_seed=1000)
    assert len(comparison["runs"]) == 6
    labels = {r.scenario for r in comparison["runs"]}
    assert labels == {"clustered", "random"}
    seeds_by_scenario = {
        name: sorted(r.seed for r in res["runs"])
        for name, res in comparison["scenarios"].items()
    }
    # Same replicate count and seed set shared across scenarios (common
    # random numbers) -- this is what makes the comparison controlled.
    assert seeds_by_scenario["clustered"] == seeds_by_scenario["random"] == [1000, 1001, 1002]


def test_write_scenario_comparison_csv_round_trips(tmp_path):
    scenarios = {
        "clustered": _base_config(clustered_cities=(0,), random_chance=0.1),
        "random": _base_config(),
    }
    comparison = run_scenario_comparison(scenarios, num_runs=2, base_seed=2000)
    path = tmp_path / "comparison.csv"
    write_scenario_comparison_csv(comparison, str(path))
    assert path.exists()

    import csv
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["scenario"] for row in rows} == {"clustered", "random"}
    # Raw output must be enough to recompute a summary later without rerunning.
    assert all(row["attack_rate"] != "" for row in rows)


def test_write_run_results_csv_single_scenario(tmp_path):
    result = run_experiment(_base_config(), num_runs=3, base_seed=10,
                            experiment_name="smoke_test")
    path = tmp_path / "runs.csv"
    write_run_results_csv(result["runs"], str(path))
    import csv
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all(row["experiment_name"] == "smoke_test" for row in rows)
