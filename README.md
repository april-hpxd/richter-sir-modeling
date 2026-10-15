# Computational Modeling of Regional Disease Spread

A **fully data-driven multi-city SEIR epidemic simulator** with configurable
contact networks, heterogeneous populations, asymmetric travel matrices and
multi-day trips. Disease spreads within cities through contact networks (seeded
random graphs, Watts–Strogatz, or well-mixed) and between cities via temporary
traveler movement. Everything is data-driven: change only the configuration to
test 2, 3, 5, 10, or 20 cities with different populations and travel patterns
without modifying code.

### Intra-City Spread
People in each city are modeled as a **contact network**:
- Each **node** is one individual.
- Each **edge** is a recurring social contact.
- Disease spreads **only along edges**, following SEIR dynamics
  (**S**usceptible → **E**xposed → **I**nfectious → **R**ecovered).

Five contact models are available:
- **Random Network** (default): Each person assigned 1–7 persistent contacts.
- **Watts–Strogatz**: Small-world network with local clustering and shortcuts.
- **Well-Mixed**: Homogeneous mixing (validation only).
- **Daily Random**: A new bounded set of uniformly random contacts each day.
- **Clustered**: A new bounded set of contacts each day, with a configurable
  per-contact probability of selecting someone from the same cluster.

In the daily models, a contact is one distinct transmission opportunity on one
day. The contact count is drawn uniformly from the configured inclusive bounds;
partners are then redrawn the next day. Cluster membership is persistent, but
the partner is not. `within_cluster_contact_probability` applies to each
contact slot, not to the population, so it changes locality without changing
contact quantity. All draws use the simulation's seeded NumPy generator.

### Inter-City Spread
Cities are connected by a **fully configurable travel layer**:
- A **travel matrix** specifies the daily probability that an eligible resident
  of city *i* visits city *j* (can be asymmetric).
- **Multi-day trips**: travellers can stay 1, 3, 7, or custom-configured days.
- While away, travellers interact with the **destination's network** (not random
  mixing) and their disease progresses naturally.
- **Bidirectional transmission**: infectious visitors expose destination residents;
  susceptible visitors can be infected and carry disease home.
- Each trip is temporary; residents return home with any infections acquired.

### Configuration-Driven Everything
- Any reasonable number of cities should work (either 1, 2 or even 20)
- Different city sizes (100, 500, 1500 people, etc.)
- Asymmetric travel (e.g. high A→B, low B→A)
- Multi-day trips (1, 3, 7, or custom distributions)
- Different visualization modes (auto/network/cluster/heatmap/pie)
- Repeated experiments (mean/std over seeds)
- Full parameter sensitivity sweeps, exported to CSV

### Adaptive Visualization
The regional animation picks a rendering strategy from population size --
manually overridable with `--visualization-mode`:

| Mode | When (auto) | What it shows |
|---|---|---|
| `network` | city ≤150 people | Every individual as a node; edges are real contacts. |
| `cluster` | city ≤1000 people | Each city's social communities (detected from its own contact graph) as sized, coloured bubbles -- local spread stays visible without drawing every node. |
| `heatmap` | city >1000 people | A block of population tiles per city; each fixed tile (5% by default) is coloured by its own % infectious. |
| `pie` | manual only | Each city's S/E/I/R composition as an animated pie chart. |

Dashed arrows animate between *any* pair of cities with travel that day (not
just neighbours in the layout) and recent travel-caused transmissions are
drawn as fading directional strings from the source individual to the newly
infected one -- both help explain *why* an outbreak just appeared somewhere new.

All configuration flows through a single immutable `Config` object. Use:
- **CLI flags** for simple changes: `--city-populations 500,200,1500`
- **JSON files** for complex setups: heterogeneous sizes, asymmetric matrices,
  multi-day trip distributions

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.9+ with `numpy`, `networkx` and `matplotlib`.

## Quick Start

### Single-City SEIR
```bash
python main.py --single-city --save-gif epidemic.gif
```

### Regional Multi-City (Homogeneous)
```bash
python main.py --regional --number-of-cities 3 --population-per-city 100 \
  --save-gif regional.gif --visualization-mode network
```

### Heterogeneous Cities (CLI)
```bash
python main.py --regional --city-populations 500,200,1500 \
  --travel-fraction 0.4 --daily-travel-rate 0.15 \
  --visualization-mode heatmap --heatmap-tile-percent 5 --save-gif heatmap.gif
```

### Full Configuration (JSON)
```bash
python main.py --regional --config my_config.json --save-gif output.gif
```

Example `my_config.json`:
```json
{
  "city_populations": [500, 200, 1500, 75],
  "travel_matrix": [
    [0, 0.10, 0.02, 0.01],
    [0.08, 0, 0.05, 0.02],
    [0.01, 0.03, 0, 0.04],
    [0.05, 0.02, 0.03, 0]
  ],
  "trip_duration_distribution": [[1, 0.6], [3, 0.3], [7, 0.1]],
  "visualization_mode": "heatmap",
  "contact_model": "random-network",
  "infection_probability": 0.06,
  "initial_infected": 3,
  "simulation_days": 200
}
```

### Repeated Experiments
```bash
python main.py --regional --number-of-cities 2 \
  --experiment 100 --experiment-base-seed 0 --quiet
```

Runs 100 independent simulations (seeds 0–99), reports mean/std for:
- Average outbreak arrival delay
- Peak infections
- Attack rate
- Imported infections
- Others

### Decision Support Analysis
```bash
python main.py --regional --number-of-cities 3 --population-per-city 80 \
  --decision-support --decision-support-runs 3 --quiet
```

The new analysis workflow evaluates interventions such as city isolation,
travel reduction, connection removal and threshold-triggered quarantine,
then ranks them by expected reduction in regional infections and provides a
network-based summary of outbreak sources and transmission hubs.

### Sensitivity Analysis (CSV output)
```bash
python main.py --regional --population-per-city 60 --simulation-days 100 \
  --sensitivity-config mobility_grid.json --sensitivity-runs-per-combo 5 \
  --sensitivity-csv results.csv
```

`mobility_grid.json` maps any `Config` field to a list of values to sweep --
the full Cartesian product is run, every individual run's parameters and
outcomes become one CSV row (ready for a research figure):
```json
{
  "daily_travel_rate": [0.0, 0.05, 0.1, 0.2, 0.3],
  "number_of_cities": [2, 5, 10]
}
```
The same seeds are reused across every grid point (common random numbers), so
differences between rows reflect the swept parameter, not seed noise.

## Parameters

### Disease Model
| Parameter | Default | Meaning |
|---|---|---|
| `--infection-probability` | 0.06 | Per-contact transmission probability. |
| `--incubation-days` | 2 | Days in EXPOSED state. |
| `--infectious-days` | 6 | Days in INFECTIOUS state. |
| `--initial-infected` | 2 | Cases seeded in city 0. Only used by the `fixed` `--initial-infection-policy` (the default). |
| `--initial-infection-policy` | fixed | `fixed` uses `--initial-infected` literally (unchanged from before this option existed). `population-aware` instead derives a bounded (1-5), population-size-dependent count and ignores `--initial-infected` — see below. |
| `--simulation-days` | 120 | Maximum days to simulate. |

### Contact Network
| Parameter | Default | Meaning |
|---|---|---|
| `--contact-model` | random-network | `random-network`, `well-mixed`, or `watts-strogatz`. |
| `--random-degree-min` | 1 | Min contacts per node (random-network). |
| `--random-degree-max` | 7 | Max contacts per node (random-network). |
| `--watts-strogatz-k` | 8 | Mean degree (watts-strogatz). |
| `--watts-strogatz-p` | 0.1 | Rewiring probability (watts-strogatz). |
| `--daily-contacts` | 8 | Contacts per day (well-mixed only). |
| `--daily-contacts-min/max` | 1 / 7 | Inclusive daily-contact bounds for daily-random and clustered models. |
| `--cluster-count` | 4 | Alias for `--num-clusters`. |
| `--within-cluster-contact-probability` | 0.9 | Per-contact probability of selecting within the person's cluster. |
| `--clustered-cities` | — | Comma-separated regional city indices that use clustered contacts. |

### Research examples

```bash
# Regular daily mixing control
python main.py --single-city --contact-model daily-random \
  --daily-contacts-min 2 --daily-contacts-max 7

# Strong and weak clustering
python main.py --single-city --contact-model clustered --cluster-count 10 \
  --within-cluster-contact-probability 0.90
python main.py --single-city --contact-model clustered --cluster-count 10 \
  --within-cluster-contact-probability 0.50

# Repeated regular-versus-clustered scenarios (change the config between runs)
python main.py --single-city --contact-model clustered --cluster-count 5 \
  --within-cluster-contact-probability 0.90 --experiment 100 \
  --experiment-csv clustered.csv

# Three unequal cities: A and C clustered, B regular
python main.py --regional --city-populations 100,200,150 \
  --clustered-cities 0,2 --cluster-count 10 \
  --within-cluster-contact-probability 0.90
```

Compare scenarios using the same population, daily-contact bounds, disease
parameters, travel parameters, and seed set. Report peak infectious count and
day, attack rate, total infections, epidemic duration, clusters reached, and
first between-cluster transmission. For regional runs also report first
arrival/imported infections, City B outbreak status, City B peak and attack
rate, and cities reached. A city that is never infected remains `-1` for its
arrival day and is retained in per-run CSV output; aggregate delay summaries
exclude those undefined delays and report their contributing sample size.

For repeated runs, the experiment report provides mean, sample standard
deviation, and a normal-approximation 95% confidence interval. Inspect outcome
distributions before choosing tests: use a suitable multi-group test such as
ANOVA or Kruskal-Wallis for several locality levels, and check assumptions or
use permutation/bootstrap methods when they are not met. The implementation
does not assume that higher clustering must reduce transmission.

### Regional Structure
| Parameter | Meaning |
|---|---|
| `--config PATH` | Load entire configuration from JSON (overrides all CLI). |
| `--number-of-cities` | Number of cities (if `--city-populations` not set). |
| `--population-per-city` | Population per city (if `--city-populations` not set). |
| `--city-populations` | Comma-separated list: `500,200,1500`. |
| `--travel-fraction` | Eligible commuters (0–0.5). |
| `--daily-travel-rate` | Fraction of eligible who travel each day. |

### Behavioral Response and Isolation
| Parameter | Default | Meaning |
|---|---|---|
| `--behavioral-response` | off | Enable contact reduction while infectious. |
| `--behavioral-response-factor` | 0.5 | Fraction of normal contacts kept once infectious. |
| `--isolation-enabled` | off | Enable automatic city isolation. |
| `--isolation-threshold` | 0.5 | Infectious fraction that triggers isolation. |
| `--isolation-travel-multiplier` | 0.0 | Travel-matrix multiplier for an isolated city. |
| `--isolation-contact-multiplier` | 1.0 | Extra in-city contact multiplier while isolated. |

### Vaccination
| Parameter | Default | Meaning |
|---|---|---|
| `--vaccination-rate` | 0.0 | Fraction of each city's population vaccinated before the outbreak begins. `0` disables vaccination. |
| `--vaccination-strategy` | random | How vaccinated individuals are chosen: `random` (uniformly at random) or `high-degree`/`high_degree` (see below). |
| `--cluster-vaccination-rates` | — | Optional per-cluster rates (e.g. `"0,0.2,0.6"`), replacing `--vaccination-rate` for the run. Requires `--contact-model clustered` (or a clustered city in a regional run) and exactly one rate per cluster. See [Cluster-Specific Vaccination](#cluster-specific-vaccination). |

See [Vaccination](#vaccination) below for what this models and what it doesn't.

### Experiments
| Parameter | Meaning |
|---|---|
| `--experiment N` | Run N independent simulations. |
| `--experiment-base-seed` | First seed (others: +1, +2, ...). |
| `--experiment-csv PATH` | Where every `--experiment` run + aggregates are written. |
| `--sensitivity-config PATH` | JSON grid of `Config` fields to sweep (Cartesian product). |
| `--sensitivity-runs-per-combo` | Seeds run per grid point (default 5). |
| `--sensitivity-csv PATH` | Where every run's params + outcomes are written. |
| `--travel-rate-sweep` | Compare a range of daily travel rates. |
| `--travel-rates` | Comma-separated rates for `--travel-rate-sweep` (default `0,0.05,0.1,0.15,0.2`). |
| `--travel-rate-sweep-runs` | Seeds run per rate (default 5). |
| `--travel-rate-sweep-csv PATH` | Where every travel-rate-sweep run is written. |
| `--validate` | Run the validation suite and print a pass/fail report. |
| `--vaccination-coverage-sweep` | Run the vaccination dose-response experiment (see [Vaccination Coverage Experiment](#vaccination-coverage-experiment)). |
| `--vaccination-rates` | Comma-separated rates to sweep (default `0.00, 0.05, 0.10, 0.20, 0.40, 0.60`, 6 points). Accepts percents too (e.g. `0,10,20`); if every value is `> 1` the list is divided by 100. |
| `--vaccination-sweep-runs` | Replicates per rate (default 150; use e.g. `5` for a quick test). |
| `--vaccination-sweep-base-seed` | First seed of the shared seed set reused at every rate. |
| `--vaccination-sweep-csv PATH` | Where every individual replicate is written. |
| `--vaccination-sweep-summary-csv PATH` | Where the one-row-per-rate summary is written. |
| `--vaccination-sweep-plot PATH` | Optional dose-response plot (attack rate / major-outbreak / invasion probability vs rate, all with 95% CI). |
| `--major-outbreak-threshold` | `attack_rate` (fraction of total population ever infected) cutoff for a "major outbreak" (default `0.20`). This analysis's own definition, not a universal epidemiological constant. |
| `--vaccination-sweep-max-days` | Safety cap on simulated days per sweep run (default 500), independent of `--simulation-days`; a run is flagged `duration_censored` if still active (E+I>0) when this cap is hit. |

### Visualization
| Parameter | Default | Meaning |
|---|---|---|
| `--visualization-mode` | auto | `auto`, `network` (nodes), `cluster` (communities), `heatmap` (population tiles), `pie` (S/E/I/R). |
| `--heatmap-tile-percent` | 5 | Approximate share of one city represented by a heatmap square. |
| `--layout` | grid | Node layout: `grid` or `circle`. |
| `--save-gif PATH` | — | Save animation to GIF. |
| `--save-curves PATH` | — | Save SEIR curves to PNG. |
| `--interval-ms` | 400 | Milliseconds per animation frame. |
| `--export-csv PATH` | — | Export the day-by-day aggregate history to CSV. |
| `--export-node-csv PATH` | — | Export a per-individual, per-day CSV (see below). |

### Reproducibility
| Parameter | Default | Meaning |
|---|---|---|
| `--random-seed` | 42 | Seed for all randomness. |

## Vaccination

An optional, pre-outbreak intervention, disabled by default. It exists to let
experiments compare an outbreak with and without a baseline level of immunity
already in the population, not to model a real vaccine's behavior.

- **Optional**: a plain `python main.py ...` run is completely unaffected —
  vaccination only activates when `--vaccination-rate` is set above `0`, and
  when it's off the simulation doesn't even draw any extra randomness, so
  every existing run/experiment/CSV is reproduced bit-for-bit.
- **Current vaccine effectiveness is 100%**: a vaccinated individual can
  never become infected and, since they can never become infectious, can
  never transmit either.
- **Vaccination happens before the simulation begins.** A configurable
  fraction of each city's population (`--vaccination-rate`, e.g. `0.1` for
  10%) is chosen from the still-fully-susceptible population, before the
  initial cases are seeded and before day 0 of the simulation. Nobody is
  vaccinated mid-outbreak.
- **Two selection strategies** (`--vaccination-strategy`):
  - `random` (default): each eligible individual is equally likely to be
    chosen.
  - `high-degree`/`high_degree`: vaccinates the individuals with the most
    neighbors in the *pre-outbreak* contact network — i.e. targets
    individuals with the greatest number of contacts in the pre-outbreak
    contact network. Degree is computed once, before the epidemic starts,
    and never recalculated as the epidemic progresses: this is a one-time,
    static network-based intervention strategy *in the simulation*, not a
    claim about biological or real-world vaccination optimality, and not
    adaptive targeting, contact tracing, or outbreak-time vaccination. Ties
    at the cutoff degree are broken with a seeded random draw (reproducible
    given the same `--random-seed`, not dependent on node iteration order).
    Only available for network-based contact models (`random-network`,
    `watts-strogatz`, `clustered`, `daily-random`) — `well-mixed` has no
    persistent network/degree and raises an error if combined with
    `high-degree`.
- **Vaccinated individuals cannot be infected or transmit the disease.**
  They occupy a fifth disease state, `V` (alongside `S`/`E`/`I`/`R`), and stay
  `V` for the entire run — there is no waning immunity and no path back to
  `S`, `I`, or `R`.
- **How to enable it**: add `--vaccination-rate RATE` (0–1) to any
  single-city or regional run, e.g.:
  ```bash
  python main.py --single-city --vaccination-rate 0.3 --save-gif out.gif
  python main.py --regional --number-of-cities 2 --vaccination-rate 0.2 --save-curves curves.png
  ```
- **What the output records**: the console summary prints whether
  vaccination was used, how many people were vaccinated, and the resulting
  coverage, for the run as a whole (single-city) and per city plus a
  regional total (regional runs). The same figures are available
  programmatically from `Simulation.vaccination_report()` /
  `City.vaccination_report()` and from `RegionalSimulation.regional_summary()`
  (`vaccination_reports`, `total_vaccinated`, `regional_vaccination_coverage`).
  The day-by-day CSV export (`--export-csv`) gets a `vaccinated` column, and
  the SEIR curve plot and network animation show vaccinated individuals in a
  distinct color (only when the run actually used vaccination).

**This is a simplified intervention model for controlled simulation
experiments, not a realistic representation of how vaccines work.** It does
not model waning immunity, booster doses, partial effectiveness, vaccine
hesitancy, demographic targeting, vaccination campaigns during an active
outbreak, contact tracing, or multiple vaccine types — those are possible
future extensions, not part of this milestone.

### Cluster-Specific Vaccination

An optional, additional layer on top of the vaccination feature above: instead
of one uniform rate across the whole population, each cluster of the
`clustered` contact model can get its *own* vaccination rate. It is entirely
opt-in — a run that doesn't pass `--cluster-vaccination-rates` behaves exactly
as before (uniform `--vaccination-rate`, or no vaccination at all).

- **"Cluster-specific", not necessarily "geographic".** The `clustered`
  contact model's clusters are local neighbourhoods/communities in the
  contact network (see [Contact Network](#contact-network) above) — they are
  not tied to any geographic coordinate. This feature lets
  you vaccinate one community more than another (e.g. modeling differential
  uptake or differential targeting across communities), not "vaccinate this
  physical region." A regional (multi-city) run's *cities* are a separate,
  coarser notion of place; cluster-specific rates apply *within* a single
  clustered city's internal clusters.
- **How to enable it**: pass `--cluster-vaccination-rates` as a comma-separated
  list with exactly one rate (0–1) per cluster, ordered by cluster id, e.g.
  for `--num-clusters 3`:
  ```bash
  python main.py --single-city --population-size 2000 --contact-model clustered \
      --num-clusters 3 --cluster-vaccination-rates "0,0.2,0.6" \
      --vaccination-strategy random --simulation-days 60
  ```
  This disables cluster 0, vaccinates ~20% of cluster 1, and ~60% of cluster 2
  — each computed from *that cluster's own population*, not the whole city's.
  Setting `--cluster-vaccination-rates` makes `--vaccination-rate` irrelevant
  for that run (the uniform path is not used at all).
- **Selection within each cluster** uses the same `--vaccination-strategy` as
  ordinary vaccination (`random` or `high-degree`/`high_degree`), but applied
  *independently inside each cluster* — there is no cross-cluster selection,
  nobody is ever vaccinated twice, and a cluster's count can never exceed its
  own population. `high-degree` ranks each cluster's members by degree in the
  full pre-outbreak contact graph (computed once and reused, not recomputed
  per cluster — see the performance note below), restricted to that cluster's
  member ids.
- **Per-cluster counts use the same rounding as uniform vaccination**:
  `round(rate * cluster_population)`.
- **Overall coverage is a weighted total, not the average of the rates.**
  Because clusters can be different sizes, `overall_vaccination_rate` is
  computed as `total_vaccinated / total_population`, which is *not* the same
  as the arithmetic mean of the per-cluster rates when clusters are
  unequal-sized. For example, cluster sizes `(50, 100, 200)` with rates
  `(0, 0.2, 0.6)` vaccinate `(0, 20, 120)` people, for an overall rate of
  `140 / 350 ≈ 40%` — not the `(0 + 0.2 + 0.6) / 3 ≈ 27%` a naive mean would
  suggest.
- **What the output records**: `Simulation.vaccination_report()` (and
  `City.vaccination_report()`) gains `cluster_vaccination_rates` (the
  requested rates), `vaccinated_count_per_cluster` and
  `population_per_cluster` (both dicts keyed by cluster id), plus
  `vaccinated_count` and `overall_vaccination_rate` (present for every run,
  cluster-specific or not — aliases of the existing `number_vaccinated` /
  `vaccination_coverage`). The console summary prints a per-cluster
  breakdown underneath the overall coverage line. Visualization is unchanged:
  the `clustered` contact model already lays out and colors clusters
  distinctly, and vaccinated individuals already render in their own color
  (see [Vaccination](#vaccination) above) — cluster-specific runs need no
  extra visualization work.
- **Performance**: cluster membership (`cluster_of`) is assigned once at
  network-construction time and never recomputed; degree for `high-degree`
  is likewise computed once over the whole graph (reusing the same one-time
  approach as ordinary `high-degree` vaccination) and then restricted to each
  cluster's node set, rather than rescanning the graph per cluster. This
  keeps the feature practical at populations up to ~100,000.
- **Validation**: `--cluster-vaccination-rates` must have exactly one entry
  per cluster, every entry must be in `[0, 1]`, and the contact model must be
  `clustered` (or the relevant regional city must be in `--clustered-cities`)
  — any other case raises a clear `ValueError` rather than silently
  coercing or ignoring the option.
- **Not implemented** (same scope limits as ordinary vaccination, plus):
  outbreak-time/adaptive vaccination, contact tracing, traveler-specific
  vaccination, vaccine waning/partial efficacy/multiple vaccine types,
  automatic/ML-based allocation across clusters, and changing cluster
  membership.

### Vaccination Coverage Experiment

A dose-response experiment: run many replicates at each of several
vaccination rates and compare outcomes across rates. Builds entirely on the
existing experiment-runner machinery (`experiments.py`'s `RunResult` and
`mean_and_ci`) — no new simulation behavior.

**Quick test** (small, fast — run this before the full sweep):
```bash
python main.py --regional --number-of-cities 2 --population-per-city 50 \
  --vaccination-coverage-sweep --vaccination-sweep-runs 5
```

**Full 0–60% experiment** (6 rates × 150 replicates = 900 simulations —
budget real time for this; see Performance below):
```bash
python main.py --regional --number-of-cities 2 --population-per-city 50 \
  --vaccination-coverage-sweep --vaccination-sweep-runs 150 \
  --vaccination-sweep-csv vaccination_runs.csv \
  --vaccination-sweep-summary-csv vaccination_summary.csv \
  --vaccination-sweep-plot vaccination_dose_response.png
```
Omit `--vaccination-rates` to use the default 6 points (`0.00, 0.05, 0.10,
0.20, 0.40, 0.60`); pass a comma-separated list (e.g.
`--vaccination-rates 0,0.1,0.2,0.3`, or equivalently as percents,
`--vaccination-rates 0,10,20,30` — if every value given is `> 1` the whole
list is treated as percents and divided by 100, so a bare `5` always means
5% and never 500%) to sweep a different set. `--vaccination-sweep-runs` is
fully configurable (100/150/200/250/... or any value). Every rate reuses the
same `--vaccination-sweep-base-seed .. + --vaccination-sweep-runs - 1` seed
set (common random numbers), so differences between rates reflect the swept
coverage, not seed noise, and the sweep is reproducible: the same config +
rates + run count + base seed always reproduce the same per-run results.
`--vaccination-sweep-max-days` (default `500`) is an independent safety cap
on simulated days per run, overriding `--simulation-days` for the sweep only
— every run continues until no `EXPOSED`/`INFECTIOUS` individuals remain
(true extinction) or this cap is hit (in which case it is flagged
`duration_censored`), not just until `INFECTIOUS` reaches 0 while `EXPOSED`
is still nonzero.

**Definitions used in the output:**
- **`vaccination_rate`**: fraction of each city's population vaccinated
  before the outbreak begins (see [Vaccination](#vaccination)).
- **Major outbreak**: `attack_rate >= 0.20` for that replicate (the
  `--major-outbreak-threshold` default, now expressed as a *fraction of
  total population* rather than a raw infection count — a fixed head-count
  threshold meant something different at every population size, which was a
  real bug fixed in the reliability/correctness milestone; see
  `CHANGELOG_RESEARCH.md`). `attack_rate = total_infected / total_population`
  where `total_infected` counts only individuals ever `EXPOSED`/`INFECTIOUS`/
  recovered-from-infection (never vaccinated-but-uninfected people, never
  double-counted) and `total_population` sums every city's full population
  (regional: across all cities). This 20% cutoff is **this analysis's own
  chosen definition, not a universal epidemiological constant** — change it
  explicitly via `--major-outbreak-threshold` if you want a different one; it
  is never changed silently.
- **Invasion probability**: the fraction of *all* replicates at a rate in
  which the outbreak reached at least one city beyond the seed city
  (`cities_reached > 1` — "City-2 invasion" in the 2-city setup). A run
  where the seeded cases burned out locally counts as a non-invasion (`0`),
  not as missing data.
- **Arrival delay (`mean_arrival_delay_successful`)**: the mean days-to-
  arrival, computed **only** over replicates that actually invaded
  (`successful_invasions` gives that count) — a non-invading replicate has
  no arrival day and is excluded, left as `NaN` rather than coerced to `-1`
  or `0`, so this and every other conditional metric always reports the
  sample size (`n`) it was actually computed over.

**Termination metadata** (per replicate, in `--vaccination-sweep-csv`):
alongside `duration_censored` (unchanged, already derived from the true
final day's state), each row also has `final_exposed`, `final_infectious`
(the last recorded day's counts), `extinction_reached` (`True` iff both are
`0`), `actual_days_run` (the day the run actually stopped on),
`simulation_max_days` (the configured cap, for context), `termination_reason`
(`"extinction"` or `"max_days"`), and `major_outbreak` (the `attack_rate >=
0.20` flag, precomputed so raw and summary agree by construction). These are
purely additive, read-only-auditable fields — no stopping/counting logic
changed when they were added (see `CHANGELOG_RESEARCH.md`). A row's
`extinction_reached` and `duration_censored` always disagree (one or the
other), and `extinction_reached=True` implies `final_exposed ==
final_infectious == 0`.

**Data-quality audit**: `python audit_experiment.py <runs.csv>
[<summary.csv>] --expected-rates 0.00 0.05 0.10 0.20 0.40 0.60
--expected-replicates 150 --expected-population 2000 --expected-cities 2`
checks replicate/seed structure, population/city counts,
`attack_rate`/`major_outbreak` arithmetic, raw-vs-summary agreement
(major-outbreak and censored counts), `cities_reached`/arrival-delay
invariants, and termination-field consistency, printing a PASS/FAIL report
naming every failed check and the affected seeds/rates. A check against a
CSV that predates the termination-metadata columns above is reported as
"cannot be checked" rather than silently passed or failed.

**Output files:**
- `--vaccination-sweep-csv` (default `vaccination_coverage_runs.csv`): one
  row per replicate, every `RunResult` field (identification, model,
  network, and outcome columns — same schema as `--experiment-csv`), so any
  summary statistic can be recomputed later without rerunning.
- `--vaccination-sweep-summary-csv` (default
  `vaccination_coverage_summary.csv`): one row per vaccination rate —
  `n_runs`, mean/median attack rate with 95% (normal-approximation) CI, mean
  peak infectious/epidemic duration with CI (descriptive only — see the
  censoring caveat below), invasion probability with a **95% Wilson score
  CI** (not a normal-approximation CI — the normal approximation can produce
  a zero-width or out-of-range interval exactly when a rate has 0 or all
  qualifying runs, which happens routinely at high coverage), major-outbreak
  count/percent/Wilson CI/median attack rate, plus a major-outbreaks-only
  breakdown (`major_outbreak_mean_*`) and censoring visibility (`n_censored`,
  `censored_fraction` — see `epidemic_stats.is_duration_censored`; a
  censored run's duration **and** its recorded attack rate are lower bounds,
  not final/true values, and are never silently averaged in as if they were
  final).
- `--vaccination-sweep-plot` (optional): a 3-panel PNG — mean attack rate,
  major-outbreak probability, and regional invasion probability, each against
  vaccination coverage, all three with 95% CI error bars, explicit axis
  units/sample size (`n` replicates/rate) in the title, the 20%-threshold
  panel noting it is this analysis's own definition, and any rate with
  >=10% of its runs censored annotated directly on the attack-rate panel.

**Performance**: each replicate is one full regional simulation; the full
6-rate × 150-run experiment is 900 simulations (a 5-run smoke test on this
machine took a few seconds per rate — scale that up to estimate your own
runtime before launching the full sweep). No parallelism is used (none
exists elsewhere in this project), so start with a small
`--vaccination-sweep-runs` to validate your configuration first.

**Methodological note**: this experiment compares outcomes across
vaccination coverage levels for *this simulated population and contact
structure*. It does not establish a real-world vaccination threshold, and
the infection probability swept elsewhere in this project is a per-contact
transmission probability, not a validated estimate of a basic reproduction
number (`R0`) — see `Config.estimated_r0()` for the simulator's own
explicitly-approximate formula, used nowhere in this experiment's analysis.

## Architecture

```
config.py              Data-driven configuration (immutable, JSON-loadable)
disease_model.py       SEIR states, individuals, per-day PersonSnapshot
engine.py              Disease progression, transmission, behavioral response
interaction.py         Contact model interface + 3 implementations
city.py                Independent city with engine, network, history
travel.py              Travel layer: matrix-based, multi-day trips, isolation
regional_simulation.py Regional coordinator (no disease, no travel logic)
epidemic_stats.py      Summary analysis (read-only)
node_export.py         Per-individual, per-day CSV export
validation.py          Validation-suite checks + pass/fail report
visualization.py       Animations (4 modes, auto-selected) + curves
experiments.py         Repeated-simulation + sensitivity/travel-rate sweeps
analysis.py            Manual what-if intervention comparison (decision support)
main.py                CLI entry point + wiring
```

### Key Classes

**Config**: Fully data-driven, JSON-loadable.
- City sizes can be per-city or uniform.
- Travel matrix is configurable, asymmetric.
- Trip duration distribution is configurable.
- Resolvers (`city_sizes()`, `travel_probability_matrix()`) compute derived values.

**City**: Independent SEIR simulation (no hardcoding for "City A" or "City B").
- Owns its own engine, network, RNG and history.
- Knows nothing about other cities.
- Can run completely standalone or as part of RegionalSimulation.

**TravelManager**: Pure travel logic (no disease knowledge).
- Builds commuter pools once from master RNG.
- Drives multi-day trips per configurable matrix.
- Delegates disease progression to City.

**RegionalSimulation**: Thin coordinator.
- Daily: advance disease in all cities → execute travel → record stats.
- No disease logic, no travel logic—just sequencing.
- Scales to any number of cities and population sizes.

## Statistics

**Per-city**: peak infectious/exposed/recovered (+day), attack rate, epidemic
duration, first infection day, imported infections (cases this city acquired via
travel), exported infections (cases this city's residents/travellers caused
elsewhere) and more.

**Regional**: total regional infections, cities reached, average arrival delay,
infection sources (which city seeded each outbreak), travel events and more.

**Experiments**: mean and standard deviation of headline metrics over repeated runs.

**Sensitivity analysis**: every individual run's swept parameters + outcomes as
one CSV row, for building research figures outside the simulator.

### Behavioral Response and City Isolation
```bash
python main.py --regional --number-of-cities 2 --population-per-city 80 \
  --behavioral-response --behavioral-response-factor 0.5 \
  --isolation-enabled --isolation-threshold 0.5 --quiet
```
- `--behavioral-response` halves (configurable via `--behavioral-response-factor`)
  an individual's daily contacts once they become infectious -- applied
  identically to residents and to hosted travelers.
- `--isolation-enabled` automatically isolates a city once its infectious
  share crosses `--isolation-threshold` (default 50%). Isolation scales that
  city's travel row/column by `--isolation-travel-multiplier` (default 0 = no
  travel) and its in-city contacts by `--isolation-contact-multiplier`.
  Isolation is one-way (a city that isolates stays isolated) and independent
  per city.

### Node-Level Per-Individual Export
```bash
python main.py --regional --number-of-cities 3 --population-per-city 100 \
  --daily-travel-rate 0.1 --export-node-csv nodes.csv --quiet
```
Writes one row per person per simulated day: `day, person_id, home_city,
current_city, state, days_in_state, traveling, infected_by,
infection_generation, infection_day, recovery_day, contacts_today,
newly_infected`. `infected_by` is a global id (`"{city}-{person}"`) that works
across the travel boundary, so a full transmission tree can be reconstructed
after the run. `--single-city --export-node-csv PATH` writes the same schema
for a single-city run.

### Batch Experiments (CSV + confidence intervals)
```bash
python main.py --regional --number-of-cities 2 \
  --experiment 100 --experiment-base-seed 0 --experiment-csv experiment.csv --quiet
```
Runs 100 independent simulations (seeds 0-99), reports mean/std/95% CI for
average outbreak arrival delay, peak infections, attack rate, imported
infections and more and writes every individual run plus the aggregates to
`--experiment-csv`.

### Travel-Rate Comparison
```bash
python main.py --regional --number-of-cities 2 --population-per-city 60 \
  --travel-rate-sweep --travel-rates "0,0.05,0.1,0.15,0.2" \
  --travel-rate-sweep-runs 5 --travel-rate-sweep-csv travel_rates.csv --quiet
```
A convenience wrapper over the generic sensitivity sweep: runs each travel
rate `--travel-rate-sweep-runs` times and prints/exports a comparison table
of arrival day, peak infections, attack rate and epidemic duration per rate.

### Validation Suite
```bash
python main.py --validate
```
Runs a fast suite of correctness invariants (zero infection probability, zero
travel, same-seed reproducibility, different-seed variability, small/large
populations, varying city counts) and prints a pass/fail report. The same
checks are mirrored as pytest tests in `tests/test_validation.py`.

### Decision Support Analysis
```bash
python main.py --regional --number-of-cities 3 --population-per-city 80 \
  --decision-support --decision-support-runs 3 --quiet
```

The analysis workflow (`analysis.py`) evaluates manual what-if interventions
such as city isolation, travel reduction, connection removal and
threshold-triggered quarantine across repeated seeds, ranks them by expected
reduction in regional infections and reports a network-based summary of
outbreak sources and transmission hubs. This is a separate, exploratory tool
from `--isolation-enabled` (which drives isolation live, automatically,
inside a normal run).

## Statistics (expanded)

Beyond the per-city and regional metrics above, every regional run also
reports:
- **Effective reproduction number**: `regional_summary()["mean_effective_r"]`
  (and `["effective_r_by_generation"]`), estimated from the transmission
  generations tracked on every individual (`Rt(g) = cases in generation g+1
  / cases in generation g`).
- **Cities isolated**: which cities have triggered `--isolation-enabled`.

## Future Extensions

The architecture naturally supports:
- **Transportation networks**: replace matrix with routing model.
- **Seasonality**: modulate transmission probability per time-of-year.
- **Vaccination**: add immune compartments.
- **Multiple strains**: track variant-specific immunity.



clustering populations vs random populations

separate travel transmissions from the inter-city transmissions
for graphs add average from multiple simulations and peak infections between cluster vs regular and regular vs regular