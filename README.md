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

### Decision Support Analysis
```bash
python main.py --regional --number-of-cities 3 --population-per-city 80 \
  --decision-support --decision-support-runs 3 --quiet
```

The new analysis workflow evaluates interventions such as city isolation,
travel reduction, connection removal, and threshold-triggered quarantine,
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

## Complete Parameter Reference

Every adjustable parameter is listed below, grouped the same way `python
main.py --help` groups them. For each: **What** it does in the program,
**Why** you'd reach for it, **How** it actually works inside the simulation,
and a runnable example. Run `python main.py --help` at any time to see the
same list with current defaults from the terminal.

### Simulation Mode

#### `--single-city` *(flag, default: off)*
- **What:** Runs one isolated city instead of a multi-city region.
- **Why:** Use it to validate pure disease dynamics (incubation, infectious
  period, network effect) without any travel confounding the picture.
- **How:** Builds a single `Simulation`/`DiseaseEngine` over
  `--population-size` people and skips the `RegionalSimulation`/`TravelManager`
  layer entirely.
```bash
python main.py --single-city --population-size 200 --save-gif single.gif
```

#### `--regional` *(flag, default: on)*
- **What:** Runs the multi-city region (this is the default mode; you rarely
  need to type it explicitly).
- **Why:** This is what you want for anything involving travel, multiple
  cities, isolation, or cross-city comparisons.
- **How:** Builds `--number-of-cities` (or `--city-populations`) independent
  `City` objects plus a `TravelManager` that moves people between them.
```bash
python main.py --regional --number-of-cities 3 --population-per-city 100 --save-gif regional.gif
```

### Disease Model Parameters

#### `--population-size N` *(default: `50`)*
- **What:** Number of individuals in a `--single-city` run.
- **Why:** Larger populations give smoother, more statistically stable SEIR
  curves; smaller ones are faster and easier to inspect node-by-node.
- **How:** Creates exactly `N` `Individual` objects, all `SUSCEPTIBLE` except
  the seeded cases. Ignored in `--regional` mode (use `--population-per-city`
  / `--city-populations` there instead).
```bash
python main.py --single-city --population-size 500 --save-curves curves.png
```

#### `--daily-contacts N` *(default: `8`)*
- **What:** How many distinct people each person meets per day, **only**
  under the `well-mixed` contact model.
- **Why:** Controls transmission opportunity when you deliberately want
  homogeneous mixing instead of a persistent social network (e.g. to sanity
  check against textbook SEIR equations).
- **How:** `WellMixedContactModel` draws `N` uniformly random distinct others
  for every individual, every day (no persistent structure at all).
```bash
python main.py --single-city --contact-model well-mixed --daily-contacts 12
```

#### `--infection-probability P` *(default: `0.06`)*
- **What:** The chance a single contact between an infectious person and a
  susceptible person results in transmission.
- **Why:** This is the single biggest lever on how explosive an outbreak is —
  raise it to simulate a more transmissible disease, lower it to simulate
  masking/precautions.
- **How:** Every day, for every infectious individual's contacts, a
  susceptible contact is exposed with probability `P` (`rng.random() < P`).
  Combined with contact count and infectious duration it determines R0 (see
  `config.estimated_r0()`, printed in every report).
```bash
python main.py --regional --infection-probability 0.12 --number-of-cities 2
```

#### `--incubation-days N` *(default: `2`)*
- **What:** How many days someone spends `EXPOSED` (infected but not yet
  contagious) before becoming `INFECTIOUS`.
- **Why:** Models a disease's latent period; a longer incubation delays when
  an outbreak becomes visible/detectable.
- **How:** Each `EXPOSED` individual's `days_in_state` increments daily;
  once it reaches `N` they flip to `INFECTIOUS` (see `DiseaseEngine._progress`).
```bash
python main.py --regional --incubation-days 5 --infectious-days 4
```

#### `--infectious-days N` *(default: `6`)*
- **What:** How many days someone stays `INFECTIOUS` (contagious) before
  recovering.
- **Why:** Longer infectious periods mean more total transmission
  opportunities per case — a direct driver of attack rate and R0.
- **How:** Mirrors `--incubation-days` but for the `INFECTIOUS -> RECOVERED`
  transition; also sets `recovery_day` on the individual for the node export.
```bash
python main.py --regional --infectious-days 10
```

#### `--initial-infected N` *(default: `2`)*
- **What:** Number of "patient zero" cases seeded (as `EXPOSED`, not yet
  contagious) at the very start of the run.
- **Why:** More seeds mean a faster, more certain takeoff; fewer seeds (even
  1) let you study whether an outbreak dies out by chance (stochastic
  extinction) — useful with `--experiment` to see how often that happens.
- **How:** `DiseaseEngine.seed_exposed(N)` picks `N` random susceptibles in
  city 0 and exposes them on day 0; every other city starts fully
  susceptible so its "arrival day" is meaningful.
```bash
python main.py --regional --initial-infected 1 --experiment 50
```

#### `--simulation-days N` *(default: `120`)*
- **What:** The maximum number of days simulated.
- **Why:** Long enough to let the epidemic fully burn out (or cap runtime for
  quick experiments/sweeps where you only need early dynamics).
- **How:** The run loop steps at most `N` times, but stops earlier once no
  one is `EXPOSED`/`INFECTIOUS` anywhere (`is_epidemic_active()`), so a large
  `N` costs nothing once the disease has died out.
```bash
python main.py --regional --simulation-days 300
```

#### `--random-seed N` *(default: `42`)*
- **What:** The single seed that determines every random draw in the run
  (who gets infected, who travels, network structure, everything).
- **Why:** Reproducibility — the same seed with the same config always
  produces byte-for-byte identical results (see `--validate`), which is what
  lets you isolate the effect of changing one parameter at a time.
- **How:** Seeds one `numpy.random.default_rng(N)`; every city and the travel
  layer spawn their own sub-generators from it, so the whole run — network,
  contacts, transmission, travel — is one deterministic stream.
```bash
python main.py --regional --random-seed 7 --save-gif seed7.gif
```

#### `--contact-model {random-network,well-mixed,watts-strogatz,clustered}` *(default: `random-network`)*
- **What:** Chooses how people are connected within a city.
- **Why:** `random-network` (default) is a realistic persistent social graph;
  `well-mixed` is the simplest textbook assumption (good for validation);
  `watts-strogatz` adds small-world clustering with occasional long-range
  shortcuts (good for modeling tight communities with a few "bridge" people);
  `clustered` splits the city into local neighbourhoods with dense in-group
  contacts and only a trickle of cross-group contacts (see `--num-clusters`/
  `--random-chance` below, and `--clustered-cities` for the regional case).
- **How:** Set once at city construction (see `interaction.py`); the engine
  calls `contact_model.contacts(id, rng)` every day and never knows which
  concrete model it's talking to.
```bash
python main.py --single-city --contact-model watts-strogatz --watts-strogatz-k 6 --watts-strogatz-p 0.15
```

#### `--random-degree-min N` / `--random-degree-max N` *(defaults: `1` / `7`)*
- **What:** Lower/upper bound on how many persistent contacts each person
  has, under `random-network`.
- **Why:** Widening the range adds heterogeneity (some very well-connected
  "super-spreader" nodes, some nearly isolated ones); narrowing it makes
  everyone roughly equally connected.
- **How:** Each node's degree is drawn uniformly in `[min, max]`, then a
  Havel-Hakimi graph with exactly those degrees is built and randomised with
  degree-preserving edge swaps (`RandomNetworkContactModel`).
```bash
python main.py --single-city --contact-model random-network --random-degree-min 2 --random-degree-max 15
```

#### `--watts-strogatz-k N` *(default: `8`)*
- **What:** Each node's number of nearest-neighbour connections before
  rewiring, under `watts-strogatz`.
- **Why:** Roughly sets everyone's contact count (like `--daily-contacts` for
  well-mixed, but with real network structure/clustering).
- **How:** Passed straight to `networkx.watts_strogatz_graph(n, k, p)`.
```bash
python main.py --single-city --contact-model watts-strogatz --watts-strogatz-k 10
```

#### `--watts-strogatz-p P` *(default: `0.1`)*
- **What:** Probability that any given local edge gets "rewired" to a random
  long-range connection, under `watts-strogatz`.
- **Why:** `p=0` is a pure ring lattice (only local spread, slow); higher `p`
  adds long-range shortcuts that let disease jump across the network fast
  (the classic "small world" effect) — try sweeping this to see spread speed
  change dramatically.
- **How:** Passed straight to `networkx.watts_strogatz_graph(n, k, p)`.
```bash
python main.py --single-city --contact-model watts-strogatz --watts-strogatz-p 0.4
```

#### `--num-clusters N` *(default: `4`)*
- **What:** Number of local clusters/neighbourhoods a `clustered` city's
  population is split into.
- **Why:** Fewer, larger clusters approximate a handful of big communities
  (e.g. neighbourhoods); more, smaller clusters approximate tight households
  or friend groups. Only affects cities actually using the `clustered` model.
- **How:** People are shuffled (seeded) and split into `num_clusters` groups
  as evenly as possible via `numpy.array_split` — this scales automatically
  to each city's own population, so `--city-populations 50,500` with
  `--num-clusters 5` gives ~10-person clusters in the first city and
  ~100-person clusters in the second, without any extra configuration.
```bash
python main.py --single-city --population-size 200 --contact-model clustered --num-clusters 10
```

#### `--random-chance P` *(default: `0.1`, range `[0, 1]`)*
- **What:** For a `clustered` city, the fraction of a person's persistent
  contacts that get moved to someone *outside* their own cluster.
- **Why:** This is social-contact structure, not infection or travel
  probability — it controls how "leaky" the clusters are. `0.0` is fully
  segregated clusters (disease can only escape via travel to another city);
  `1.0` removes the cluster restriction almost entirely; `0.1`-`0.3` is a
  realistic "mostly local, some outside mixing" neighbourhood.
- **How:** After building each cluster's local ring of contacts, every edge
  is rewired to a random member of a *different* cluster with probability
  `random_chance` (mirrors `watts-strogatz`'s own rewiring, just constrained
  to land outside the original cluster) -- see `ClusteredContactModel` in
  `interaction.py`.
```bash
python main.py --single-city --contact-model clustered --num-clusters 5 --random-chance 0.3
```

### Regional Simulation Parameters

#### `--config PATH` *(default: none)*
- **What:** Loads the *entire* configuration from a JSON file, ignoring every
  other CLI flag.
- **Why:** The only practical way to set heterogeneous city sizes, an
  asymmetric travel matrix, and a custom trip-duration distribution all at
  once (these can't be expressed as simple comma-separated CLI flags).
- **How:** `Config.from_json(PATH)` parses the file into a `Config`
  dataclass; unknown keys are ignored, missing keys fall back to defaults.
```bash
python main.py --regional --config my_config.json --save-gif output.gif
```

#### `--number-of-cities N` *(default: `2`)*
- **What:** Number of cities to simulate, each with population
  `--population-per-city`.
- **Why:** Study how outbreak dynamics change with more/fewer connected
  populations (more cities = more paths for spread, but each individual city
  may see a smaller share of cases).
- **How:** Ignored if `--city-populations` is set (which implies both the
  count and each city's size). Feeds `Config.city_sizes()`.
```bash
python main.py --regional --number-of-cities 6 --population-per-city 80
```

#### `--population-per-city N` *(default: `50`)*
- **What:** Population of every city, when `--city-populations` isn't given.
- **Why:** Scale the whole region up/down uniformly; larger cities give
  smoother statistics per city, smaller ones make node-level animations
  (`network` mode) more readable.
- **How:** Used by `Config.city_sizes()` to build `N` identical-size cities.
```bash
python main.py --regional --number-of-cities 4 --population-per-city 250
```

#### `--city-populations "500,200,1500"` *(default: none)*
- **What:** Explicit, comma-separated population for each city — also
  determines the number of cities.
- **Why:** Model realistic heterogeneous regions (one big metro area plus
  several small towns) instead of identical-sized cities.
- **How:** Parsed into a tuple and overrides `--number-of-cities`/
  `--population-per-city` entirely.
```bash
python main.py --regional --city-populations 500,200,1500 --daily-travel-rate 0.1
```

#### `--clustered-cities "0,2"` *(default: none)*
- **What:** Comma-separated city indices that use the `clustered` contact
  model instead of the regular `--contact-model`. Cities *not* listed keep
  using the regular network unchanged (`random-network` by default, or
  whatever `--contact-model` is set to) — you never need to list "regular"
  cities separately. Works for any number of cities and any mix of
  populations; each clustered city sizes its own clusters from its own
  population (see `--num-clusters`).
- **Why:** This is the whole point of the feature: directly compare cities
  with different social contact structure inside one regional run, e.g. "does
  a clustered City 0 seed City 1 slower than a well-mixed City 0 would?"
  without touching disease, travel, or any other parameter.
- **How:** `Config.city_contact_model_types()` resolves one contact model per
  city (`"clustered"` for indices in `clustered_cities`, else the global
  `contact_model`); `RegionalSimulation` builds each city's `ContactModel`
  from that per-city resolution. Invalid indices raise a clear `ValueError`.
```bash
# City 0 and City 2 clustered, City 1 stays on the regular random-network model
python main.py --regional --city-populations 50,100,75 --clustered-cities 0,2 \
    --num-clusters 5 --random-chance 0.1 --random-seed 42 --save-gif mixed.gif
```

#### `--travel-fraction F` *(default: `0.5`, range `0`–`0.5`)*
- **What:** The fraction of each city's population that is *eligible* to
  travel at all (the rest never leave home).
- **Why:** Models the fact that not everyone travels — commuters/travelers
  are usually a subset of the population. Lower it to represent a more
  "stay-at-home" population.
- **How:** A fixed pool is chosen once per city at the start of the run
  (`TravelManager.eligible`); only members of this pool are ever considered
  for a trip. Its size is `floor(population * F)`, with one additional member
  selected with probability equal to the fractional remainder. Thus the
  expected pool size is exactly `population * F`, without systematically
  reducing a small non-zero fraction to zero.
```bash
python main.py --regional --travel-fraction 0.2 --number-of-cities 3
```

#### `--daily-travel-rate R` *(default: `0.1`, range `0`–`1`)*
- **What:** Of the eligible travelers who are currently home, the probability
  each one travels *somewhere* today.
- **Why:** This is the main "how connected are these cities" dial — the
  parameter most people mean by "travel rate"; sweep it with
  `--travel-rate-sweep` to see how mobility affects arrival speed and total
  spread.
- **How:** Used to build a uniform `travel_matrix` (split evenly across the
  other cities) when no explicit matrix is given via `--config`; each
  eligible-and-home resident travels today with probability `R`.
```bash
python main.py --regional --daily-travel-rate 0.25 --number-of-cities 2
```

### Behavioral Response and Isolation

#### `--behavioral-response` *(flag, default: off)*
- **What:** Turns on automatic contact reduction for anyone who becomes
  infectious (people who feel sick tend to interact less).
- **Why:** Models realistic self-protective behavior (staying home when
  symptomatic) and lets you measure how much that alone slows an outbreak,
  independent of any official intervention.
- **How:** Once enabled, every infectious individual's daily contact list is
  subsampled by `--behavioral-response-factor` before transmission is
  computed (`DiseaseEngine.effective_contacts`); applies identically to
  residents and to travelers currently hosted in another city.
```bash
python main.py --regional --behavioral-response --behavioral-response-factor 0.4
```

#### `--behavioral-response-factor F` *(default: `0.5`, range `0`–`1`)*
- **What:** The fraction of *normal* contacts an infectious person keeps once
  `--behavioral-response` is on (e.g. `0.5` = half as many contacts).
- **Why:** Tune how strong the self-isolating behavior is — `1.0` would be no
  change at all, `0.1` models someone who almost fully withdraws.
- **How:** `k = round(len(contacts) * F)` contacts are randomly kept out of
  the full contact list for that day; has no effect unless
  `--behavioral-response` is also passed.
```bash
python main.py --regional --behavioral-response --behavioral-response-factor 0.25
```

#### `--isolation-enabled` *(flag, default: off)*
- **What:** Turns on automatic, live city isolation once a city's outbreak
  gets bad enough.
- **Why:** Models a real-world circuit breaker — "close the city down once
  X% are sick" — and lets you measure how much delay/containment that buys
  versus doing nothing.
- **How:** Every day, `RegionalSimulation.step()` checks each city's
  infectious fraction; once it crosses `--isolation-threshold`, that city
  flips to isolated **permanently** (one-way) and its travel/contacts are cut
  per the two multipliers below.
```bash
python main.py --regional --isolation-enabled --isolation-threshold 0.3 --number-of-cities 3
```

#### `--isolation-threshold F` *(default: `0.5`, range `0`–`1`)*
- **What:** The infectious fraction of a city's population (e.g. `0.5` =
  50%) that triggers isolation.
- **Why:** Lower it to simulate an earlier, more cautious lockdown trigger;
  raise it to simulate waiting until the outbreak is severe.
- **How:** Compared each day against `city.history[-1].infectious /
  population`; only takes effect when `--isolation-enabled` is set.
```bash
python main.py --regional --isolation-enabled --isolation-threshold 0.15
```

#### `--isolation-travel-multiplier F` *(default: `0.0`, range `0`–`1`)*
- **What:** How much of an isolated city's travel is still allowed —
  `0` means no travel in or out at all, `1` means travel is unaffected.
- **Why:** Models partial vs. total travel bans (e.g. essential travel only
  might be `0.1`, a hard border closure is `0.0`).
- **How:** `TravelManager.set_city_isolated` scales that city's entire
  travel-matrix row and column by this multiplier (relative to the original
  unisolated values, so lifting isolation would restore them exactly).
```bash
python main.py --regional --isolation-enabled --isolation-travel-multiplier 0.1
```

#### `--isolation-contact-multiplier F` *(default: `1.0`, range `0`–`1`)*
- **What:** An *extra* in-city contact reduction applied to everyone in an
  isolated city, stacked on top of any `--behavioral-response`.
- **Why:** Models a local lockdown (reduced gatherings, closed venues) on top
  of individual self-isolating behavior — the two are independent levers.
- **How:** Multiplied into the same `effective_contacts` factor used for
  behavioral response, so `1.0` (default) means isolation only restricts
  travel, not in-city mixing, unless you lower it.
```bash
python main.py --regional --isolation-enabled --isolation-contact-multiplier 0.5
```

### Experiments

#### `--experiment N` *(default: `0` = disabled)*
- **What:** Runs the same regional configuration `N` times with different
  seeds and reports mean/std/95% CI instead of a single run.
- **Why:** A single run is one random draw; real conclusions ("travel
  increases attack rate") need many seeds to separate signal from noise.
- **How:** Calls `RegionalSimulation` once per seed in
  `--experiment-base-seed .. --experiment-base-seed + N - 1`, then aggregates
  arrival delay, peak infections, attack rate, imported infections, etc.
```bash
python main.py --regional --number-of-cities 2 --experiment 100 --quiet
```

#### `--experiment-base-seed N` *(default: `0`)*
- **What:** The first seed used by `--experiment` (subsequent runs use
  `N+1, N+2, ...`).
- **Why:** Change this to run a *different* batch of seeds without
  overlapping a previous experiment's seeds.
- **How:** Passed straight through to `run_experiment(base_seed=N)`.
```bash
python main.py --regional --experiment 50 --experiment-base-seed 1000
```

#### `--experiment-csv PATH` *(default: `experiment_results.csv`)*
- **What:** Where every individual `--experiment` run (plus the aggregate
  mean/std/CI) gets written as CSV.
- **Why:** So you can re-analyze/re-plot results (e.g. in Excel or pandas)
  without rerunning the simulation.
- **How:** `write_experiment_csv` writes one row per seed, a blank line, then
  a `metric, mean, std, ci95, n` block.
```bash
python main.py --regional --experiment 100 --experiment-csv my_experiment.csv
```

#### `--sensitivity-config PATH` *(default: none)*
- **What:** Runs a full parameter sweep from a JSON file mapping any
  `Config` field name to a list of values, e.g.
  `{"daily_travel_rate": [0.0, 0.1, 0.2], "number_of_cities": [2, 5, 10]}`.
- **Why:** The general tool for "how does outcome Y change as parameter X
  varies" questions across *any* parameter (or combination of parameters),
  not just travel rate.
- **How:** Sweeps the full Cartesian product of every listed field, running
  `--sensitivity-runs-per-combo` seeds per combination (same seeds reused
  across combinations, so differences reflect the parameter, not seed luck).
```bash
python main.py --regional --sensitivity-config mobility_grid.json --sensitivity-csv results.csv
```

#### `--sensitivity-runs-per-combo N` *(default: `5`)*
- **What:** Independent seeds run per grid point in `--sensitivity-config`.
- **Why:** More seeds per point = less noisy comparison between grid points,
  at the cost of more total runs.
- **How:** Each grid point runs seeds `base_seed .. base_seed + N - 1`.
```bash
python main.py --regional --sensitivity-config mobility_grid.json --sensitivity-runs-per-combo 20
```

#### `--sensitivity-csv PATH` *(default: `sensitivity_results.csv`)*
- **What:** Where every sensitivity-sweep run (swept parameters + every
  outcome metric) is written, one row per run.
- **Why:** This is the file you'd load into a plotting tool to build a
  research figure (e.g. attack rate vs. travel rate).
- **How:** `write_sensitivity_csv` puts the swept parameter columns first,
  then `seed`, then every outcome metric.
```bash
python main.py --regional --sensitivity-config mobility_grid.json --sensitivity-csv sweep.csv
```

#### `--travel-rate-sweep` *(flag, default: off)*
- **What:** A ready-made comparison across a range of daily travel rates —
  no JSON file needed.
- **Why:** The single most common "compare travel rates" question has its
  own one-flag shortcut instead of hand-writing a `--sensitivity-config` file.
- **How:** Internally calls the same sensitivity-sweep machinery with
  `{"daily_travel_rate": rates}` and prints a table of arrival day, peak
  infections, attack rate, and duration per rate.
```bash
python main.py --regional --number-of-cities 2 --travel-rate-sweep
```

#### `--travel-rates "0,0.05,0.1,0.15,0.2"` *(default: `"0,0.05,0.1,0.15,0.2"`)*
- **What:** The comma-separated list of daily travel rates compared by
  `--travel-rate-sweep`.
- **Why:** Customize which rates you want side-by-side (e.g. finer steps
  around a suspected threshold).
- **How:** Parsed into floats and passed as the sweep values for
  `daily_travel_rate`.
```bash
python main.py --regional --travel-rate-sweep --travel-rates "0,0.02,0.04,0.06,0.08,0.1"
```

#### `--travel-rate-sweep-runs N` *(default: `5`)*
- **What:** Independent seeds run per rate in `--travel-rate-sweep`.
- **Why:** Same reasoning as `--sensitivity-runs-per-combo` — more seeds per
  rate reduces noise in the comparison table.
- **How:** Passed through as `runs_per_combo` to the underlying sweep.
```bash
python main.py --regional --travel-rate-sweep --travel-rate-sweep-runs 15
```

#### `--travel-rate-sweep-csv PATH` *(default: `travel_rate_sweep.csv`)*
- **What:** Where every individual `--travel-rate-sweep` run is written.
- **Why:** Keep the raw per-seed data, not just the printed averages table.
- **How:** Same CSV writer used by `--sensitivity-csv`.
```bash
python main.py --regional --travel-rate-sweep --travel-rate-sweep-csv rates.csv
```

#### `--validate` *(flag, default: off)*
- **What:** Instead of running a simulation, runs a fast suite of
  correctness checks and prints PASS/FAIL for each.
- **Why:** Sanity-check that the simulator itself is behaving correctly
  (e.g. after changing code, or just to build confidence in the tool) before
  trusting any research results from it.
- **How:** Runs `validation.run_all_validations()` — zero infection
  probability, zero travel, same-seed reproducibility, different-seed
  variability, tiny/huge populations, and varying city counts — and prints a
  report. All other flags are ignored when `--validate` is passed.
```bash
python main.py --validate
```

### Visualization and Output

#### `--visualization-mode {auto,network,cluster,heatmap,pie}` *(default: `auto`)*
- **What:** Chooses how the regional animation renders each city.
  `auto` picks automatically from the largest city's population:
  `network` (≤150 people: every individual as a node), `cluster` (≤1000:
  social communities as bubbles), `heatmap` (>1000: population tiles).
  `pie` (S/E/I/R composition) is manual-only.
- **Why:** The node-level `network` view is the most informative but becomes
  unreadable/slow for thousands of people — the other modes trade node-level
  detail for readability at scale.
- **How:** Dispatches to one of `animate_regional_states` /
  `animate_regional_clusters` / `animate_regional_heatmap` /
  `animate_regional_pies` in `visualization.py`.
```bash
python main.py --regional --city-populations 2000,500 --visualization-mode heatmap --save-gif big.gif
```

#### `--heatmap-tile-percent P` *(default: `5`)*
- **What:** Approximately what percent of a city's population one heatmap
  tile represents (only relevant in `heatmap` mode).
- **Why:** Smaller tiles (lower `P`) show finer-grained spatial detail within
  a city; larger tiles are faster to render for huge populations.
- **How:** `tile_count = round(1 / (P/100))`; residents are split into that
  many equal fixed cohorts, each tile coloured by its own infectious share.
```bash
python main.py --regional --city-populations 3000 --visualization-mode heatmap --heatmap-tile-percent 2
```

#### `--layout {grid,circle}` *(default: `grid`)*
- **What:** The fixed spatial arrangement of individual nodes in `network`
  mode.
- **Why:** Purely visual preference — `circle` can make travel/transmission
  lines easier to follow for small populations; `grid` scales better.
- **How:** `grid_layout`/`circle_layout` assign each person a fixed `(x, y)`
  position once; positions never move for the whole animation, only colour
  and highlights change.
```bash
python main.py --single-city --layout circle --save-gif circle.gif
```

#### `--interval-ms N` *(default: `400`)*
- **What:** Milliseconds shown per animation frame (one frame = one day).
- **Why:** Slow it down to study a specific transition day-by-day; speed it
  up for a quick overview of a long run.
- **How:** Passed straight to `matplotlib.animation.FuncAnimation(interval=N)`
  and used to derive the saved GIF's frame rate (`fps = 1000/N`).
```bash
python main.py --single-city --interval-ms 150 --save-gif fast.gif
```

#### `--save-gif PATH` *(default: none)*
- **What:** Saves the node/cluster/heatmap/pie animation to a `.gif` file.
- **Why:** GIFs are the easiest way to *see* an outbreak unfold and to share
  results (e.g. in a report or presentation).
- **How:** `FuncAnimation.save(PATH, writer=PillowWriter(...))`; also
  triggers the animation to be built even without `--show`.
```bash
python main.py --regional --number-of-cities 2 --save-gif regional.gif
```

#### `--save-curves PATH` *(default: none)*
- **What:** Saves the S/E/I/R count-over-time curves to a `.png` image.
- **Why:** The classic epidemic-curve chart — usually what you want for a
  written report rather than an animation.
- **How:** `plot_curves`/`plot_regional_curves` plot each compartment's daily
  count and save with `matplotlib.figure.savefig`.
```bash
python main.py --regional --number-of-cities 3 --save-curves curves.png
```

#### `--decision-support` *(flag, default: off)*
- **What:** After the main run, additionally evaluates a fixed set of
  what-if interventions (isolate city 0, isolate city 1, halve travel
  globally, remove the connection between cities 0 and 1) and ranks them.
- **Why:** Answers "which intervention would have helped most?" — useful for
  a policy-style discussion section in a report.
- **How:** Re-runs the region under each intervention for
  `--decision-support-runs` seeds via `analysis.evaluate_interventions`, then
  prints a ranked report plus network-centrality analysis of each city.
```bash
python main.py --regional --number-of-cities 3 --decision-support
```

#### `--decision-support-runs N` *(default: `3`)*
- **What:** Number of repeated seeds used to evaluate each intervention in
  `--decision-support`.
- **Why:** More runs give a more reliable ranking (with a real 95% CI on the
  reduction estimate) at the cost of more total simulations.
- **How:** Each intervention (and the baseline) is run `N` times; results are
  averaged with a 95% confidence interval.
```bash
python main.py --regional --decision-support --decision-support-runs 10
```

#### `--export-csv PATH` *(default: none)*
- **What:** Exports the day-by-day **aggregate** history (S/E/I/R counts per
  day) to CSV. Single-city mode only.
- **Why:** Lightweight export when you just want the compartment counts over
  time, not full per-individual detail.
- **How:** `epidemic_stats.export_csv` writes one row per day with
  `day, susceptible, exposed, infectious, recovered, new_exposed,
  new_infectious, new_recovered`.
```bash
python main.py --single-city --export-csv history.csv
```

#### `--export-node-csv PATH` *(default: none)*
- **What:** Exports a full **per-individual, per-day** CSV — every person,
  every day, with their state, who infected them, and their travel status.
  Works for both `--single-city` and `--regional` runs.
- **Why:** This is the file for real statistical analysis: survival curves,
  transmission trees, "did travelers get infected more than non-travelers,"
  etc. — anything you can't get from the aggregate counts alone.
- **How:** Columns: `day, person_id, home_city, current_city, state,
  days_in_state, traveling, infected_by, infection_generation, infection_day,
  recovery_day, contacts_today, newly_infected`. `person_id`/`infected_by`
  use a global `"{city}-{person}"` id so cross-city transmission chains are
  fully reconstructable.
```bash
python main.py --regional --number-of-cities 3 --daily-travel-rate 0.1 --export-node-csv nodes.csv
```

#### `--show` *(flag, default: off)*
- **What:** Opens interactive matplotlib windows for the animation/curves
  instead of (or in addition to) saving them to a file.
- **Why:** Quick visual check while iterating on parameters, without
  producing a file each time.
- **How:** Passes `show=True` through to the plotting/animation functions,
  which call `plt.show()`.
```bash
python main.py --single-city --show
```

#### `--quiet` *(flag, default: off)*
- **What:** Suppresses the per-day progress printout during the run.
- **Why:** Essential for `--experiment`/`--sensitivity-config`/
  `--travel-rate-sweep`, which run many simulations — without it you'd get a
  wall of per-day text for every single run.
- **How:** Simply skips the `verbose=True` per-day print in `Simulation.run`/
  `RegionalSimulation.run`; does not affect the final summary report.
```bash
python main.py --regional --experiment 100 --quiet
```

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

**Experiments**: mean, standard deviation, and 95% CI of headline metrics
over repeated runs.

**Sensitivity analysis**: every individual run's swept parameters + outcomes as
one CSV row, for building research figures outside the simulator.

**Effective reproduction number**: `regional_summary()["mean_effective_r"]`
(and the per-generation breakdown in `["effective_r_by_generation"]`),
estimated from the transmission generations tracked on every individual
(`Rt(g) = cases in generation g+1 / cases in generation g`).

**Cities isolated**: `regional_summary()["cities_isolated"]` -- which cities
have triggered `--isolation-enabled`.

See [Complete Parameter Reference](#complete-parameter-reference) above for
runnable examples of every export, sweep, and analysis flag
(`--export-node-csv`, `--experiment-csv`, `--travel-rate-sweep`,
`--validate`, `--decision-support`, `--behavioral-response`,
`--isolation-enabled`, ...).

## Future Extensions

The architecture naturally supports:
- **Transportation networks**: replace matrix with routing model.
- **Seasonality**: modulate transmission probability per time-of-year.
- **Vaccination**: add immune compartments.
- **Multiple strains**: track variant-specific immunity.
