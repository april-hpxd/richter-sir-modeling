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
    # Default sweep points per the reliability/analysis milestone:
    # 0/5/10/20/40/60% (6 points), replacing the previous 13-point (5%-step)
    # default.
    assert DEFAULT_VACCINATION_COVERAGE_RATES == (0.0, 0.05, 0.10, 0.20, 0.40, 0.60)
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


# 8. Major outbreak classification correctly uses attack_rate >= 0.20 (a
# *fraction of total population*, not a raw infection count -- this is the
# fix from the reliability/analysis milestone: a fixed head-count threshold
# silently means something different at every population size).
def test_major_outbreak_threshold():
    runs = run_vaccination_coverage_sweep(
        _base_config(infection_probability=0.9, random_degree_min=5,
                    random_degree_max=8),
        vaccination_rates=(0.0,), num_runs=10, base_seed=20)
    for run in runs:
        assert is_major_outbreak(run, threshold=0.20) == (run.attack_rate >= 0.20)
    assert DEFAULT_MAJOR_OUTBREAK_THRESHOLD == 0.20


# Boundary direction: exactly at the threshold counts as a major outbreak
# (">=", not ">"), just below does not, just above does -- regardless of the
# underlying population size (denominator is always total population).
def test_major_outbreak_boundary_direction():
    base = run_vaccination_coverage_sweep(
        _base_config(), vaccination_rates=(0.0,), num_runs=1, base_seed=0)[0]
    exactly_20 = dataclasses.replace(base, attack_rate=0.20)
    just_below = dataclasses.replace(base, attack_rate=0.1999)
    just_above = dataclasses.replace(base, attack_rate=0.2001)
    assert is_major_outbreak(exactly_20, threshold=0.20) is True
    assert is_major_outbreak(just_below, threshold=0.20) is False
    assert is_major_outbreak(just_above, threshold=0.20) is True


def test_major_outbreak_denominator_is_total_population_not_a_raw_count():
    # Two runs with the same raw infection count but different populations:
    # only the smaller population's run should qualify once the threshold is
    # expressed as a fraction of population, which is exactly the bug this
    # milestone fixes relative to a fixed raw-count threshold.
    base = run_vaccination_coverage_sweep(
        _base_config(), vaccination_rates=(0.0,), num_runs=1, base_seed=0)[0]
    small_population_run = dataclasses.replace(
        base, total_infected=30, population=100, attack_rate=30 / 100)
    large_population_run = dataclasses.replace(
        base, total_infected=30, population=5000, attack_rate=30 / 5000)
    assert is_major_outbreak(small_population_run) is True
    assert is_major_outbreak(large_population_run) is False


def test_vaccinated_but_uninfected_not_counted_toward_major_outbreak():
    # A heavily vaccinated run where the outbreak barely spreads should have
    # a small attack rate (vaccinated people are never counted as infected --
    # see city.summary_stats()/Vaccination Milestone 1) and therefore not
    # qualify as a major outbreak, even though a large fraction of the
    # population is no longer susceptible.
    runs = run_vaccination_coverage_sweep(
        _base_config(), vaccination_rates=(0.6,), num_runs=5, base_seed=5)
    for run in runs:
        # attack_rate only reflects actual infections, never vaccinated-only
        # coverage, so it must never exceed 1 - vaccination_rate (bounded by
        # the susceptible pool that could ever be infected).
        assert run.attack_rate <= 1.0 - run.vaccination_rate + 1e-9
        assert is_major_outbreak(run) == (run.attack_rate >= 0.20)


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
    assert "major_outbreak_ci_low" in rows[0]
    assert "major_outbreak_ci_high" in rows[0]


# -- Wilson CI for proportions (replaces the normal-approximation CI that
# -- mean_and_ci would otherwise give for a 0/1 indicator) --
def test_wilson_ci_stays_within_unit_interval_at_extremes():
    from analysis import wilson_ci

    low, high = wilson_ci(0, 20)  # 0% observed
    assert 0.0 <= low <= high <= 1.0
    assert low == 0.0  # Wilson's lower bound at p=0 is exactly 0

    low, high = wilson_ci(20, 20)  # 100% observed
    assert 0.0 <= low <= high <= 1.0
    assert high == pytest.approx(1.0, abs=1e-9)  # upper bound at p=1 is ~1


def test_wilson_ci_matches_known_reference_value():
    from analysis import wilson_ci

    # 10/100 successes, 95% Wilson CI (z=1.96): [0.05524, 0.17438],
    # independently derivable from the closed-form Wilson formula.
    low, high = wilson_ci(10, 100)
    assert low == pytest.approx(0.05524, abs=1e-4)
    assert high == pytest.approx(0.17438, abs=1e-4)


def test_wilson_ci_empty_sample_is_nan():
    from analysis import wilson_ci

    low, high = wilson_ci(0, 0)
    assert low != low and high != high  # NaN != NaN


def test_summary_major_outbreak_ci_is_wilson_not_normal_approximation():
    # At 0% observed major outbreaks, the normal approximation collapses to a
    # zero-width interval ([0, 0]); Wilson gives a non-degenerate interval
    # whose upper bound is strictly positive.
    runs = run_vaccination_coverage_sweep(
        _base_config(infection_probability=0.0),
        vaccination_rates=(0.0,), num_runs=10, base_seed=2)
    summary = summarize_vaccination_coverage(runs)[0]
    assert summary["major_outbreak_percent"] == 0.0
    assert summary["major_outbreak_ci_low"] == 0.0
    assert summary["major_outbreak_ci_high"] > 0.0


def test_summary_invasion_ci_is_wilson():
    runs = run_vaccination_coverage_sweep(
        _base_config(travel_fraction=0.0, daily_travel_rate=0.0),
        vaccination_rates=(0.0,), num_runs=6, base_seed=1)
    summary = summarize_vaccination_coverage(runs)[0]
    assert summary["invasion_probability"] == 0.0
    assert summary["invasion_ci_low"] == 0.0
    assert summary["invasion_ci_high"] > 0.0


# -- Regional attack-rate denominator: total population across all cities --
def test_attack_rate_denominator_is_total_regional_population():
    config = _base_config(number_of_cities=3, population_per_city=25)
    runs = run_vaccination_coverage_sweep(
        config, vaccination_rates=(0.0,), num_runs=3, base_seed=7)
    for run in runs:
        assert run.population == 75  # 3 cities x 25
        assert run.attack_rate == pytest.approx(run.total_infected / run.population)


# -- Censoring consistency: a censored run is still classified by the same
# -- attack_rate >= threshold rule as an extinct run (no special-casing). --
def test_major_outbreak_classification_consistent_for_censored_and_extinct_runs():
    # A short horizon with a fast-spreading disease produces a mix of
    # censored (still active at the cap) and extinct runs; both kinds must
    # use exactly the same attack_rate-based rule.
    runs = run_vaccination_coverage_sweep(
        _base_config(simulation_days=4, infection_probability=0.9,
                    random_degree_min=5, random_degree_max=8),
        vaccination_rates=(0.0,), num_runs=15, base_seed=40)
    censored = [r for r in runs if r.duration_censored]
    extinct = [r for r in runs if not r.duration_censored]
    for r in censored + extinct:
        assert is_major_outbreak(r) == (r.attack_rate >= 0.20)
    # Not asserting a particular split exists (environment-dependent), only
    # that whichever runs are censored follow the identical rule.
