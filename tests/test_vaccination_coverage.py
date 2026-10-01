import dataclasses

import pytest

from config import Config
from experiments import (
    DEFAULT_MAJOR_OUTBREAK_THRESHOLD,
    DEFAULT_VACCINATION_COVERAGE_RATES,
    is_major_outbreak,
    is_successful_invasion,
    run_vaccination_coverage_sweep,
    summarize_vaccination_coverage,
)


def _base_config(**overrides):
    defaults = dict(
        number_of_cities=2, population_per_city=40, simulation_days=25,
        daily_travel_rate=0.1, travel_fraction=0.3, initial_infected=2,
        infection_probability=0.3, contact_model="daily-random",
        random_degree_min=2, random_degree_max=6,
    )
    defaults.update(overrides)
    return Config(**defaults)


# 1. All 13 vaccination rates are accepted.
def test_default_vaccination_coverage_rates_has_13_points():
    assert len(DEFAULT_VACCINATION_COVERAGE_RATES) == 13
    assert DEFAULT_VACCINATION_COVERAGE_RATES[0] == 0.0
    assert DEFAULT_VACCINATION_COVERAGE_RATES[-1] == 0.60
    for rate in DEFAULT_VACCINATION_COVERAGE_RATES:
        Config(vaccination_rate=rate)  # must not raise


def test_sweep_runs_every_requested_rate():
    rates = (0.0, 0.1, 0.2, 0.3)
    runs = run_vaccination_coverage_sweep(
        _base_config(), vaccination_rates=rates, num_runs=2, base_seed=0)
    seen_rates = sorted({r.vaccination_rate for r in runs})
    assert seen_rates == list(rates)
    assert len(runs) == len(rates) * 2


# 2. 0% vaccination behaves as before (matches a plain no-vaccination run).
def test_zero_percent_rate_in_sweep_matches_plain_run():
    from regional_simulation import RegionalSimulation

    config = _base_config(random_seed=3)
    plain = RegionalSimulation(config)
    plain.run()

    runs = run_vaccination_coverage_sweep(
        config, vaccination_rates=(0.0,), num_runs=1, base_seed=3)
    assert runs[0].total_infected == plain.regional_summary()["total_infected"]
    assert runs[0].vaccination_rate == 0.0


# 3. 60% vaccination is accepted.
def test_sixty_percent_rate_is_accepted():
    runs = run_vaccination_coverage_sweep(
        _base_config(), vaccination_rates=(0.60,), num_runs=1, base_seed=0)
    assert len(runs) == 1
    assert runs[0].vaccination_rate == 0.60


# 4. Invalid rates below 0 or above 1 are rejected.
def test_invalid_rates_are_rejected():
    with pytest.raises(ValueError):
        Config(vaccination_rate=-0.1)
    with pytest.raises(ValueError):
        Config(vaccination_rate=1.1)


# 5. Replicate count is configurable.
def test_replicate_count_is_configurable():
    runs = run_vaccination_coverage_sweep(
        _base_config(), vaccination_rates=(0.1,), num_runs=7, base_seed=0)
    assert len(runs) == 7


# 6. Seed range is reproducible.
def test_seed_range_is_reproducible():
    config = _base_config()
    a = run_vaccination_coverage_sweep(
        config, vaccination_rates=(0.0, 0.2), num_runs=3, base_seed=10)
    b = run_vaccination_coverage_sweep(
        config, vaccination_rates=(0.0, 0.2), num_runs=3, base_seed=10)
    for ra, rb in zip(a, b):
        da, db = dataclasses.asdict(ra), dataclasses.asdict(rb)
        del da["scenario"], db["scenario"]  # label only, not a run outcome
        assert da == db
    assert [r.seed for r in a] == [10, 11, 12, 10, 11, 12]


# 7. One output row is generated per replicate.
def test_one_row_per_replicate():
    rates = (0.0, 0.1, 0.2)
    num_runs = 4
    runs = run_vaccination_coverage_sweep(
        _base_config(), vaccination_rates=rates, num_runs=num_runs, base_seed=0)
    assert len(runs) == len(rates) * num_runs


# 8. Major outbreak classification correctly uses total_infected >= 20.
def test_major_outbreak_threshold():
    runs = run_vaccination_coverage_sweep(
        _base_config(infection_probability=0.9, random_degree_min=5,
                    random_degree_max=8),
        vaccination_rates=(0.0,), num_runs=10, base_seed=20)
    for run in runs:
        assert is_major_outbreak(run, threshold=20) == (run.total_infected >= 20)
    assert is_major_outbreak(
        dataclasses.replace(runs[0], total_infected=20), threshold=20) is True
    assert is_major_outbreak(
        dataclasses.replace(runs[0], total_infected=19), threshold=20) is False
    assert DEFAULT_MAJOR_OUTBREAK_THRESHOLD == 20


# 9. Non-invasion runs are not included as successful arrival-delay
# observations.
def test_non_invasion_runs_excluded_from_arrival_delay():
    runs = run_vaccination_coverage_sweep(
        _base_config(travel_fraction=0.0, daily_travel_rate=0.0),
        vaccination_rates=(0.0,), num_runs=5, base_seed=0)
    assert all(not is_successful_invasion(r) for r in runs)
    summary = summarize_vaccination_coverage(runs)
    assert summary[0]["successful_invasions"] == 0
    assert summary[0]["mean_arrival_delay_successful"] != summary[0]["mean_arrival_delay_successful"]  # NaN


# 10. Invasion probability denominator is all runs.
def test_invasion_probability_denominator_is_all_runs():
    runs = run_vaccination_coverage_sweep(
        _base_config(travel_fraction=0.0, daily_travel_rate=0.0),
        vaccination_rates=(0.0,), num_runs=6, base_seed=1)
    summary = summarize_vaccination_coverage(runs)
    assert summary[0]["n_runs"] == 6
    assert summary[0]["invasion_probability"] == 0.0  # 0 invasions / 6 runs


# 11. Major-outbreak percentage denominator is all runs.
def test_major_outbreak_percent_denominator_is_all_runs():
    runs = run_vaccination_coverage_sweep(
        _base_config(infection_probability=0.0),
        vaccination_rates=(0.0,), num_runs=8, base_seed=2)
    summary = summarize_vaccination_coverage(runs)
    assert summary[0]["n_runs"] == 8
    assert summary[0]["major_outbreak_count"] == 0
    assert summary[0]["major_outbreak_percent"] == 0.0


# 12. Major-outbreak-only statistics use only qualifying runs.
def test_major_outbreak_only_statistics_use_qualifying_subset():
    runs = run_vaccination_coverage_sweep(
        _base_config(infection_probability=0.9, random_degree_min=5,
                    random_degree_max=8, population_per_city=60),
        vaccination_rates=(0.0,), num_runs=20, base_seed=30)
    summary = summarize_vaccination_coverage(runs)[0]
    qualifying = [r for r in runs if is_major_outbreak(r)]
    assert summary["major_outbreak_count"] == len(qualifying)
    if qualifying:
        import numpy as np
        expected_median = float(np.median([r.attack_rate for r in qualifying]))
        assert summary["major_outbreak_median_attack_rate"] == pytest.approx(expected_median)


def test_summary_has_one_row_per_rate_sorted_ascending():
    runs = run_vaccination_coverage_sweep(
        _base_config(), vaccination_rates=(0.3, 0.0, 0.15), num_runs=2, base_seed=0)
    summary = summarize_vaccination_coverage(runs)
    assert [row["vaccination_rate"] for row in summary] == [0.0, 0.15, 0.3]
    assert all(row["n_runs"] == 2 for row in summary)


def test_write_vaccination_coverage_csvs(tmp_path):
    from experiments import (
        write_vaccination_coverage_runs_csv,
        write_vaccination_coverage_summary_csv,
    )
    runs = run_vaccination_coverage_sweep(
        _base_config(), vaccination_rates=(0.0, 0.1), num_runs=2, base_seed=0)
    runs_path = tmp_path / "runs.csv"
    write_vaccination_coverage_runs_csv(runs, str(runs_path))
    assert runs_path.exists()

    summary = summarize_vaccination_coverage(runs)
    summary_path = tmp_path / "summary.csv"
    write_vaccination_coverage_summary_csv(summary, str(summary_path))
    assert summary_path.exists()

    import csv
    with open(summary_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert "vaccination_rate" in rows[0]
    assert "invasion_probability" in rows[0]
    assert "major_outbreak_percent" in rows[0]
