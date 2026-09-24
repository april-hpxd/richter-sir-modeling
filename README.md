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
| `--initial-infected` | 2 | Cases seeded in city 0. |
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