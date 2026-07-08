# Computational Modeling of Regional Disease Spread — Milestone 1

A **single-city network SIR epidemic simulator**. This is the first milestone
of a larger Rcihter Project that will eventually simulate
influenza spreading between Midwestern cities connected by rail travel. In this
milestone I want to build the reusable disease engine for one city; later milestones
will instantiate it many times and add an inter-city mobility layer.

## What it should do

Rather than the classical homogeneous-mixing SIR model (where everyone can
infect everyone), people are modeled as a **contact network**:

- Each **node** is one individual.
- Each **edge** is a recurring social contact.
- Disease spreads **only along edges**, following SIR dynamics
  (**S**usceptible → **I**nfected → **R**ecovered), where recovery grants
  permanent immunity.

The network is a **Watts–Strogatz small-world graph**: dense local clustering
(neighborhoods, schools, workplaces, families) plus a few long-range shortcuts
(occasional outside contact). This should be much more realistic than random mixing.

## Requirements (As of now)

```
pip install -r requirements.txt
```

Python 3.9+ with `networkx`, `numpy`, `scipy`, and `matplotlib`. Saving
animations as **MP4** additionally needs an `ffmpeg` binary on your `PATH`;
**GIF** export works out of the box.


## Planned Parameters

All parameters need to live in [`config.py`] and are exposed on the command line:

| Parameter | CLI flag | Meaning |
|---|---|---|
| `population_size` | `--population-size` | Number of individuals (nodes). |
| `average_contacts` | `--average-contacts` | Mean contacts per person (even). |
| `rewiring_probability` | `--rewiring-probability` | Small-world rewiring (β). |
| `infection_probability` | `--infection-probability` | Per-contact, per-day transmission. |
| `recovery_probability` | `--recovery-probability` | Per-day recovery (1/mean infectious period). |
| `initial_infected` | `--initial-infected` | Number of patient zeros. |
| `simulation_days` | `--simulation-days` | Maximum days to simulate. |
| `random_seed` | `--random-seed` | Seed for full reproducibility. |
| `animation_speed` | `--animation-speed` | Milliseconds per frame. |

A useful rule to use for statistics:
`R0 ~= infection_probability * average_contacts / recovery_probability`

## Planned Architecture

Each file should have one job:

```
config.py             # Config: all tunable parameters (immutable)
network_generator.py  # Build the Watts–Strogatz network + fixed layout
sir_engine.py         # SIREngine: the core day-by-day SIR dynamics (reusable)
simulation.py         # Simulation: orchestrates a run + records history
statistics.py         # Summaries, R estimate, live panel text, CSV export
visualization.py      # Network + geographic animations, summary plots
main.py               # Command-line entry point wiring it all together
```

The **`SIREngine`** and **`Simulation`** classes are to be self-contained: 
each owns its own network, RNG stream and state, with no
global dependencies and no coupling to visualization.

## Path to the multi-city model

This milestone is designed to become one reusable city object:

```
Simulation (this milestone)
    --  
City class  (owns its own Config, network, engine, history)
    --  
Multiple City objects
    --
Rail mobility layer  (moves infected individuals between cities)
    --
Regional epidemic simulation
```
