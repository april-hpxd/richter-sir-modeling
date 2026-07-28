# Computational Modeling of Regional Disease Spread

A **fully data-driven multi-city SEIR epidemic simulator** with configurable
contact networks, heterogeneous populations, asymmetric travel matrices, and
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

Three contact models are available:
- **Random Network** (default): Each person assigned 1–7 persistent contacts.
- **Watts–Strogatz**: Small-world network with local clustering and shortcuts.
- **Well-Mixed**: Homogeneous mixing (validation only).

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
just neighbours in the layout), and recent travel-caused transmissions are
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

Python 3.9+ with `numpy`, `networkx`, and `matplotlib`.

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

### Regional Structure
| Parameter | Meaning |
|---|---|
| `--config PATH` | Load entire configuration from JSON (overrides all CLI). |
| `--number-of-cities` | Number of cities (if `--city-populations` not set). |
| `--population-per-city` | Population per city (if `--city-populations` not set). |
| `--city-populations` | Comma-separated list: `500,200,1500`. |
| `--travel-fraction` | Eligible commuters (0–0.5). |
| `--daily-travel-rate` | Fraction of eligible who travel each day. |

### Experiments
| Parameter | Meaning |
|---|---|
| `--experiment N` | Run N independent simulations. |
| `--experiment-base-seed` | First seed (others: +1, +2, ...). |
| `--sensitivity-config PATH` | JSON grid of `Config` fields to sweep (Cartesian product). |
| `--sensitivity-runs-per-combo` | Seeds run per grid point (default 5). |
| `--sensitivity-csv PATH` | Where every run's params + outcomes are written. |

### Visualization
| Parameter | Default | Meaning |
|---|---|---|
| `--visualization-mode` | auto | `auto`, `network` (nodes), `cluster` (communities), `heatmap` (population tiles), `pie` (S/E/I/R). |
| `--heatmap-tile-percent` | 5 | Approximate share of one city represented by a heatmap square. |
| `--layout` | grid | Node layout: `grid` or `circle`. |
| `--save-gif PATH` | — | Save animation to GIF. |
| `--save-curves PATH` | — | Save SEIR curves to PNG. |
| `--interval-ms` | 400 | Milliseconds per animation frame. |

### Reproducibility
| Parameter | Default | Meaning |
|---|---|---|
| `--random-seed` | 42 | Seed for all randomness. |

## Architecture

```
config.py              Data-driven configuration (immutable, JSON-loadable)
disease_model.py       SEIR states and individuals
engine.py              Disease progression + transmission logic
interaction.py         Contact model interface + 3 implementations
city.py                Independent city with engine, network, history
travel.py              Travel layer: matrix-based, multi-day trips
regional_simulation.py Regional coordinator (no disease, no travel logic)
epidemic_stats.py      Summary analysis (read-only)
visualization.py       Animations (4 modes, auto-selected) + curves
experiments.py         Repeated-simulation + sensitivity-analysis framework
main.py                CLI entry point + wiring
```

### Key Classes

**Config**: Fully data-driven, JSON-loadable.
- City sizes can be per-city or uniform.
- Travel matrix is configurable, asymmetric.
- Trip duration distribution is configurable.
- Resolvers (`city_sizes()`, `travel_probability_matrix()`) compute derived values.

**City**: Independent SEIR simulation (no hardcoding for "City A" or "City B").
- Owns its own engine, network, RNG, and history.
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
elsewhere), and more.

**Regional**: total regional infections, cities reached, average arrival delay,
infection sources (which city seeded each outbreak), travel events, and more.

**Experiments**: mean and standard deviation of headline metrics over repeated runs.

**Sensitivity analysis**: every individual run's swept parameters + outcomes as
one CSV row, for building research figures outside the simulator.

## Future Extensions

The architecture naturally supports:
- **Transportation networks**: replace matrix with routing model.
- **Seasonality**: modulate transmission probability per time-of-year.
- **Vaccination**: add immune compartments.
- **Multiple strains**: track variant-specific immunity.
- **Isolation**: cities start closing after specific number of infections.

