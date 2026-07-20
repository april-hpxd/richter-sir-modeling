# Computational Modeling of Regional Disease Spread

A **multi-city SEIR epidemic simulator** with configurable contact networks and
inter-city travel. Disease spreads within cities through contact networks
(Watts-Strogatz small-world graphs) and between cities via temporary traveler
movement.

### Intra-City Spread
People in each city are modeled as a **contact network**:
- Each **node** is one individual.
- Each **edge** is a recurring social contact.
- Disease spreads **only along edges**, following SEIR dynamics
  (**S**usceptible → **E**xposed → **I**nfectious → **R**ecovered).

The network is a **Watts–Strogatz small-world graph**: dense local clustering
(neighborhoods, schools, workplaces, families) plus a few long-range shortcuts.
Alternatively, use a **well-mixed model** (any person can contact any other) for
validation or comparison.

### Inter-City Spread
Multiple independent cities are connected by a **simple commuting travel layer**:
1. Each city has a fixed pool of eligible commuters (`travel_fraction` of its residents).
2. Each day, `daily_travel_rate` of that pool makes a day trip to a random other city.
3. A traveler attaches to a random resident and mingles with that resident's
   contacts in the **destination city's own contact network** (not random mixing).
4. Transmission during the visit is **bidirectional** and the traveler's disease
   state travels with them:
   - an **infectious** visitor can expose susceptible residents (seeding the destination), and
   - a **susceptible** visitor can be infected by infectious residents and carries it home.
5. Everyone returns home at day's end; imported infections are folded into that
   same day's per-city statistics and animation frame.

Because only **City A** is seeded, this cleanly measures how long it takes for
infection to first reach **City B** (and how much its epidemic curve is delayed
relative to City A's). With the defaults, infection typically reaches City B
around day 8–13 and City B's peak trails City A's by roughly two weeks.

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.9+ with `numpy`, `networkx`, and `matplotlib`. All dependencies are
in `requirements.txt`. GIF animations are built-in; MP4 export requires
`ffmpeg` on your `PATH`.

## Quick Start

### Single-City SEIR 
```bash
python main.py --single-city --population-size 50 --save-gif single_city.gif
```

### Regional Multi-City 
```bash
python main.py --regional --number-of-cities 2 --population-per-city 50 --travel-fraction 0.5 --daily-travel-rate 0.1 --save-gif regional.gif --save-curves curves.png
```

## Parameters

### Disease Model
| Parameter | Default | Meaning |
|---|---|---|
| `--infection-probability` | 0.06 | Per-contact transmission probability. |
| `--incubation-days` | 2 | Days in EXPOSED state. |
| `--infectious-days` | 6 | Days in INFECTIOUS state. |
| `--initial-infected` | 2 | Number of initial cases in City A (EXPOSED). |
| `--simulation-days` | 120 | Maximum days to simulate. |

### Contact Network
| Parameter | Default | Meaning |
|---|---|---|
| `--daily-contacts` | 8 | Contacts per person per day (well-mixed model only). |
| `--contact-model` | watts-strogatz | Model: `well-mixed` or `watts-strogatz`. |
| `--watts-strogatz-k` | 8 | Neighborhood size (mean degree) in W-S networks. |
| `--watts-strogatz-p` | 0.1 | Rewiring probability in W-S networks. |

### Regional Simulation (Milestone 2)
| Parameter | Default | Meaning |
|---|---|---|
| `--number-of-cities` | 2 | Number of independent cities. |
| `--population-per-city` | 50 | Population in each city. |
| `--travel-fraction` | 0.5 | Fraction of population eligible to travel. |
| `--daily-travel-rate` | 0.1 | Fraction of eligible who actually travel. |

### Reproducibility
| Parameter | Default | Meaning |
|---|---|---|
| `--random-seed` | 42 | Seed for all randomness. |

### Output
| Parameter | Meaning |
|---|---|
| `--save-gif PATH` | Save state animation to GIF. |
| `--save-curves PATH` | Save SEIR curves to PNG. |
| `--layout` | Grid (`grid`) or circle (`circle`) layout. |
| `--interval-ms` | Milliseconds per animation frame. |
| `--quiet` | Suppress per-day progress output. |

## Architecture

Each module has one responsibility:

```
config.py                # Immutable configuration (all parameters)
disease_model.py         # SEIR state and Individual (no dynamics)
interaction.py           # ContactModel interface + implementations
engine.py                # DiseaseEngine: SEIR progression + transmission
simulation.py            # Simulation: orchestrates single-city run
city.py                  # City: independent city with own engine + network
regional_simulation.py   # RegionalSimulation: coordinates cities + travel
statistics.py            # Summary statistics from history
visualization.py         # Single-city and regional animations + curves
main.py                  # CLI entry point
```

### Key Classes

**DiseaseEngine**: Core SEIR dynamics
- Owns population of Individuals
- Advances disease state each day
- Uses a ContactModel to determine interactions
- Completely independent of visualization or travel logic

**City**: One city in a regional simulation
- Owns its own DiseaseEngine and contact network
- Owns its own independent RNG
- Owns its history (daily records and state frames)
- Can run completely standalone or as part of RegionalSimulation

**RegionalSimulation**: Coordinates multiple cities
- Owns a list of City objects and the fixed per-city commuter pools
- Each day: advance disease in all cities, execute travel, record statistics
- Travel model: select today's commuters, assign destinations, run
  bidirectional transmission through the destination's network, fold any
  imported infections into the affected city's day, return travelers home
- All travel randomness is drawn from the master RNG (cities keep their own
  RNGs for internal dynamics), so a single seed reproduces the whole region

## Design Principles

1. **Modularity**: Disease engine knows nothing about travel, visualization, or
   networking infrastructure. Swapping the contact model or travel layer requires
   no changes to core epidemic logic.

2. **Reproducibility**: Single master RNG seed (via `--random-seed`) drives all
   randomness. Identical seeds produce identical results.

3. **Independence**: Each City owns its own RNG (spawned from the master seed),
   network, and population. Cities can run in isolation or coordinated regionally.

4. **Immutability**: Config is frozen at construction; history is append-only.
   Visualization and statistics read from history without modifying it.

5. **Type Safety**: Extensive type hints and dataclasses prevent silent errors.
