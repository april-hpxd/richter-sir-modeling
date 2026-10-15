"""Data-quality audit for vaccination-coverage-sweep CSV output.

Reusable checker for the raw per-run CSV (`RunResult` rows, written by
``experiments.write_vaccination_coverage_runs_csv``) and its paired summary
CSV (written by ``experiments.write_vaccination_coverage_summary_csv``).

Every check either passes, fails (with the affected rows/rates named), or is
reported as "cannot check: missing column X" when the raw CSV predates a
field this audit wants to verify -- it never silently skips a check and
reports it as a pass.

Usage::

    python audit_experiment.py vaccination_runs.csv vaccination_summary.csv \\
        --expected-rates 0.00 0.05 0.10 0.20 0.40 0.60 \\
        --expected-replicates 150 --expected-population 2000 \\
        --expected-cities 2

Exits with status 0 if every checkable item passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import csv
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

MAJOR_OUTBREAK_THRESHOLD = 0.20


class AuditReport:
    """Accumulates PASS/FAIL/SKIP results for one audit run."""

    def __init__(self) -> None:
        self.results: List[tuple] = []  # (status, name, detail)

    def ok(self, name: str, detail: str = "") -> None:
        self.results.append(("PASS", name, detail))

    def fail(self, name: str, detail: str) -> None:
        self.results.append(("FAIL", name, detail))

    def missing(self, name: str, detail: str) -> None:
        self.results.append(("MISSING", name, detail))

    @property
    def passed(self) -> bool:
        """No FAIL results. MISSING (uncheckable) results do not fail the run
        -- they are reported separately so a caller can see what could not
        be verified, without a stale dataset looking like a clean pass."""
        return all(status != "FAIL" for status, _, _ in self.results)

    @property
    def has_missing(self) -> bool:
        return any(status == "MISSING" for status, _, _ in self.results)

    def print_report(self) -> None:
        print("=" * 70)
        print("DATA-QUALITY AUDIT REPORT")
        print("=" * 70)
        for status, name, detail in self.results:
            marker = {"PASS": "PASS   ", "FAIL": "FAIL   ",
                      "MISSING": "MISSING"}[status]
            print(f"[{marker}] {name}")
            if detail:
                print(f"           {detail}")
        n_pass = sum(1 for s, _, _ in self.results if s == "PASS")
        n_fail = sum(1 for s, _, _ in self.results if s == "FAIL")
        n_miss = sum(1 for s, _, _ in self.results if s == "MISSING")
        print("-" * 70)
        print(f"{n_pass} passed, {n_fail} failed, {n_miss} could not be "
              f"checked (missing columns)")
        print("=" * 70)


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _parse_first_infection_days(value: str) -> List[float]:
    try:
        parsed = ast.literal_eval(value)
        return [float(v) for v in parsed]
    except (ValueError, SyntaxError):
        return []


def audit(raw_rows: List[Dict[str, str]],
          summary_rows: List[Dict[str, str]],
          report: AuditReport,
          expected_rates: Optional[Sequence[float]] = None,
          expected_replicates: Optional[int] = None,
          expected_population: Optional[int] = None,
          expected_cities: Optional[int] = None) -> None:
    """Run every check against one raw+summary CSV pair, recording results."""
    if not raw_rows:
        report.fail("raw CSV non-empty", "raw CSV has zero rows")
        return

    columns = set(raw_rows[0].keys())

    def has(col: str) -> bool:
        return col in columns

    # -- 1. Each requested rate present exactly once in summary --
    if expected_rates is not None:
        summary_rates = [float(r["vaccination_rate"]) for r in summary_rows]
        missing = [r for r in expected_rates
                   if not any(abs(r - s) < 1e-9 for s in summary_rates)]
        dupes = [rate for rate, n in Counter(summary_rates).items() if n > 1]
        if missing or dupes:
            report.fail("summary has each expected rate exactly once",
                        f"missing={missing} duplicated={dupes}")
        else:
            report.ok("summary has each expected rate exactly once")
    else:
        report.ok("summary rate presence",
                  "no --expected-rates given; skipped by request (not a failure)")

    # -- 2. Exact expected replicate count per rate --
    by_rate: Dict[float, List[Dict]] = defaultdict(list)
    for row in raw_rows:
        by_rate[float(row["vaccination_rate"])].append(row)
    if expected_replicates is not None:
        bad_counts = {rate: len(rows) for rate, rows in by_rate.items()
                      if len(rows) != expected_replicates}
        if bad_counts:
            report.fail("expected replicate count per rate",
                        f"expected {expected_replicates}; got {bad_counts}")
        else:
            report.ok("expected replicate count per rate",
                      f"{expected_replicates} per rate x {len(by_rate)} rates")

    # -- 3. Unique seeds/replicate ids within each rate --
    dup_seeds = {}
    for rate, rows in by_rate.items():
        seeds = [r["seed"] for r in rows]
        dups = [s for s, n in Counter(seeds).items() if n > 1]
        if dups:
            dup_seeds[rate] = dups
    if dup_seeds:
        report.fail("unique seeds within each rate", f"duplicates: {dup_seeds}")
    else:
        report.ok("unique seeds within each rate")

    # -- 4. Correct total population / city count --
    if expected_population is not None:
        bad_pop = [r["seed"] for r in raw_rows
                   if int(float(r["population"])) != expected_population]
        if bad_pop:
            report.fail("population matches expected",
                        f"expected {expected_population}; "
                        f"{len(bad_pop)} rows differ (seeds: {bad_pop[:10]}...)")
        else:
            report.ok("population matches expected", str(expected_population))
    if expected_cities is not None:
        if not has("first_infection_days"):
            report.missing("city count matches expected",
                           "no 'first_infection_days' column to count cities from")
        else:
            bad_cities = []
            for r in raw_rows:
                days = _parse_first_infection_days(r["first_infection_days"])
                if len(days) != expected_cities:
                    bad_cities.append(r["seed"])
            if bad_cities:
                report.fail("city count matches expected",
                            f"expected {expected_cities}; "
                            f"{len(bad_cities)} rows differ")
            else:
                report.ok("city count matches expected", str(expected_cities))

    # -- 5. attack_rate == total_infected / population --
    bad_attack = []
    for r in raw_rows:
        pop = float(r["population"])
        expected = float(r["total_infected"]) / pop if pop else 0.0
        if abs(expected - float(r["attack_rate"])) > 1e-6:
            bad_attack.append(r["seed"])
    if bad_attack:
        report.fail("attack_rate == total_infected / population",
                    f"{len(bad_attack)} rows mismatch (seeds: {bad_attack[:10]})")
    else:
        report.ok("attack_rate == total_infected / population")

    # -- 6. major_outbreak flags match the 0.20 cutoff --
    if not has("major_outbreak"):
        report.missing("major_outbreak flag matches 0.20 cutoff",
                       "raw CSV has no 'major_outbreak' column (predates this "
                       "audit's fields) -- cannot verify per-row; see "
                       "summary-vs-raw count check below instead")
    else:
        bad_major = []
        for r in raw_rows:
            expected_flag = float(r["attack_rate"]) >= MAJOR_OUTBREAK_THRESHOLD
            if _to_bool(r["major_outbreak"]) != expected_flag:
                bad_major.append(r["seed"])
        if bad_major:
            report.fail("major_outbreak flag matches 0.20 cutoff",
                        f"{len(bad_major)} rows mismatch (seeds: {bad_major[:10]})")
        else:
            report.ok("major_outbreak flag matches 0.20 cutoff")

    # -- 7. summary counts/probabilities agree with raw data --
    if summary_rows:
        mismatches = []
        for srow in summary_rows:
            rate = float(srow["vaccination_rate"])
            rows = by_rate.get(rate, [])
            n = len(rows)
            raw_major = sum(1 for r in rows
                            if float(r["attack_rate"]) >= MAJOR_OUTBREAK_THRESHOLD)
            summary_major = int(float(srow["major_outbreak_count"]))
            if raw_major != summary_major:
                mismatches.append(
                    f"rate={rate}: raw={raw_major} summary={summary_major}")
            raw_censored = sum(1 for r in rows if _to_bool(r["duration_censored"]))
            summary_censored = int(float(srow.get("n_censored", raw_censored)))
            if raw_censored != summary_censored:
                mismatches.append(
                    f"rate={rate}: censored raw={raw_censored} "
                    f"summary={summary_censored}")
        if mismatches:
            report.fail("summary counts agree with raw data", "; ".join(mismatches))
        else:
            report.ok("summary counts agree with raw data (major_outbreak, censoring)")
    else:
        report.missing("summary counts agree with raw data", "no summary rows given")

    # -- 8. actual_days_run consistent with duration/termination metadata --
    if not all(has(c) for c in
               ("actual_days_run", "final_exposed", "final_infectious",
                "extinction_reached", "termination_reason")):
        report.missing(
            "actual_days_run consistent with termination metadata",
            "raw CSV lacks one or more of actual_days_run / final_exposed / "
            "final_infectious / extinction_reached / termination_reason -- "
            "this dataset predates these fields and cannot be audited at "
            "this granularity (duration_censored alone is still present and "
            "separately checked above/below)")
    else:
        bad_term = []
        for r in raw_rows:
            censored = _to_bool(r["duration_censored"])
            extinct = _to_bool(r["extinction_reached"])
            fe = float(r["final_exposed"])
            fi = float(r["final_infectious"])
            reason = r["termination_reason"]
            if censored == extinct:
                bad_term.append(f"seed {r['seed']}: duration_censored={censored} "
                                f"extinction_reached={extinct} (must disagree)")
                continue
            if extinct and (fe != 0 or fi != 0):
                bad_term.append(f"seed {r['seed']}: extinction_reached=True but "
                                f"final_exposed={fe} final_infectious={fi}")
            if not extinct and (fe == 0 and fi == 0):
                bad_term.append(f"seed {r['seed']}: censored but final E=I=0")
            expected_reason = "extinction" if extinct else "max_days"
            if reason != expected_reason:
                bad_term.append(f"seed {r['seed']}: termination_reason="
                                f"{reason!r}, expected {expected_reason!r}")
        if bad_term:
            report.fail("extinction/censoring/termination metadata consistent",
                        f"{len(bad_term)} issues: {bad_term[:10]}")
        else:
            report.ok("extinction/censoring/termination metadata consistent")

    # -- 9. arrival delays only over valid arrivals (no -1 sentinel averaged) --
    if not has("average_arrival_delay"):
        report.missing("arrival delay excludes no-arrival sentinel",
                       "no 'average_arrival_delay' column")
    else:
        sentinel_included = [r["seed"] for r in raw_rows
                             if float(r["average_arrival_delay"]) < 0
                             and float(r["cities_reached"]) > 1]
        if sentinel_included:
            report.fail("arrival delay excludes no-arrival sentinel",
                        f"rows with cities_reached>1 but negative delay: "
                        f"{sentinel_included[:10]}")
        else:
            report.ok("arrival delay excludes no-arrival sentinel")
        if summary_rows:
            for srow in summary_rows:
                if "mean_arrival_delay_successful" in srow:
                    val = srow["mean_arrival_delay_successful"]
                    if val not in ("", "nan") and float(
                            srow.get("successful_invasions", 0)) == 0:
                        if val.lower() != "nan":
                            report.fail(
                                "zero-invasion rate has no fabricated arrival delay",
                                f"rate={srow['vaccination_rate']}: "
                                f"successful_invasions=0 but "
                                f"mean_arrival_delay_successful={val}")
            else:
                report.ok("zero-invasion rates report delay as n/a (NaN), not 0")

    # -- 10. no missing/impossible values in critical fields --
    critical = ["population", "total_infected", "attack_rate",
                "epidemic_duration", "cities_reached"]
    impossible = []
    for r in raw_rows:
        pop = float(r["population"])
        ti = float(r["total_infected"])
        ar = float(r["attack_rate"])
        if ti < 0 or ti > pop:
            impossible.append(f"seed {r['seed']}: total_infected={ti} pop={pop}")
        if ar < 0 or ar > 1.0 + 1e-9:
            impossible.append(f"seed {r['seed']}: attack_rate={ar}")
        for c in critical:
            if r.get(c, "") == "":
                impossible.append(f"seed {r['seed']}: missing {c}")
    if impossible:
        report.fail("no missing/impossible values in critical fields",
                    f"{len(impossible)} issues: {impossible[:10]}")
    else:
        report.ok("no missing/impossible values in critical fields")

    # -- 11. cities_reached / invasion sanity: source city always counts,
    #        never double-counts a destination --
    if has("first_infection_days"):
        bad_invasion = []
        for r in raw_rows:
            days = _parse_first_infection_days(r["first_infection_days"])
            if not days:
                continue
            reached = sum(1 for d in days if d >= 0)
            if int(float(r["cities_reached"])) != reached:
                bad_invasion.append(
                    f"seed {r['seed']}: cities_reached={r['cities_reached']} "
                    f"but {reached} cities have day>=0")
            # Source city (index 0) must always show day 0 (always "reached").
            if days[0] != 0:
                bad_invasion.append(
                    f"seed {r['seed']}: source city first_infection_day="
                    f"{days[0]}, expected 0")
        if bad_invasion:
            report.fail("cities_reached matches count of day>=0 cities",
                        f"{len(bad_invasion)} issues: {bad_invasion[:10]}")
        else:
            report.ok("cities_reached matches count of day>=0 cities "
                      "(source city always counted)")
    else:
        report.missing("cities_reached / invasion sanity",
                       "no 'first_infection_days' column")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_csv")
    parser.add_argument("summary_csv", nargs="?")
    parser.add_argument("--expected-rates", type=float, nargs="*", default=None)
    parser.add_argument("--expected-replicates", type=int, default=None)
    parser.add_argument("--expected-population", type=int, default=None)
    parser.add_argument("--expected-cities", type=int, default=None)
    args = parser.parse_args(argv)

    raw_rows = _read_csv(args.raw_csv)
    summary_rows = _read_csv(args.summary_csv) if args.summary_csv else []

    report = AuditReport()
    audit(raw_rows, summary_rows, report,
          expected_rates=args.expected_rates,
          expected_replicates=args.expected_replicates,
          expected_population=args.expected_population,
          expected_cities=args.expected_cities)
    report.print_report()
    if not report.passed:
        return 1
    if report.has_missing:
        return 2  # no failures, but some checks were uncheckable
    return 0


if __name__ == "__main__":
    sys.exit(main())
