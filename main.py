"""Command-line entry point for the SEIR regional multi-city simulation.

Examples

Single city with defaults::

    python main.py --single-city --save-gif epidemic.gif --save-curves curves.png

Regional simulation (2 cities)::

    python main.py --regional --number-of-cities 2 --save-gif regional.gif

Two cities with travel::

    python main.py --regional --number-of-cities 2 --travel-fraction 0.5 --daily-travel-rate 0.1 --save-gif travel.gif
    
Three cities with heterogeneous populations::

    python main.py --regional --city-populations 500,200,1500 --travel-fraction 0.4 --daily-travel-rate 0.15 --visualization-mode heatmap --save-gif heatmap.gif --save-curves curves.png
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import json

from config import Config
from experiments import print_experiment_report, run_experiment, run_sensitivity_analysis
from regional_simulation import RegionalSimulation
from simulation import Simulation
from epidemic_stats import export_csv, summary
import visualization
from visualization import city_label


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser.

    Every :class:`~config.Config` field is exposed as an option defaulting to
    the value in :class:`Config`, so ``python main.py`` alone is valid.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    d = Config()
    p = argparse.ArgumentParser(
        description="SEIR epidemic simulation: single-city or regional multi-city.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = p.add_argument_group("simulation mode")
    mode.add_argument("--single-city", action="store_true",
                      help="Run a single-city simulation (ignores regional options).")
    mode.add_argument("--regional", action="store_true", default=True,
                      help="Run a regional multi-city simulation (default).")

    model = p.add_argument_group("disease model parameters")
    model.add_argument("--population-size", type=int, default=d.population_size,
                       help="Number of individuals in a single-city simulation.")
    model.add_argument("--daily-contacts", type=int, default=d.daily_contacts,
                       help="Contacts each infectious person makes per day.")
    model.add_argument("--infection-probability", type=float,
                       default=d.infection_probability,
                       help="Per-interaction transmission probability.")
    model.add_argument("--incubation-days", type=int, default=d.incubation_days,
                       help="Days exposed (incubating) before infectious.")
    model.add_argument("--infectious-days", type=int, default=d.infectious_days,
                       help="Days infectious before recovery.")
    model.add_argument("--initial-infected", type=int,
                       default=d.initial_infected,
                       help="Number of initial cases (seeded as exposed).")
    model.add_argument("--simulation-days", type=int, default=d.simulation_days,
                       help="Maximum number of days to simulate.")
    model.add_argument("--random-seed", type=int, default=d.random_seed,
                       help="Seed for all randomness (reproducibility).")
    model.add_argument("--contact-model", choices=("random-network", "well-mixed", "watts-strogatz"),
                       default=d.contact_model,
                       help="Contact network model.")
    model.add_argument("--random-degree-min", type=int,
                       default=d.random_degree_min,
                       help="Minimum persistent contacts per node in a random network.")
    model.add_argument("--random-degree-max", type=int,
                       default=d.random_degree_max,
                       help="Maximum persistent contacts per node in a random network.")
    model.add_argument("--watts-strogatz-k", type=int, default=d.watts_strogatz_k,
                       help="Neighbourhood size for Watts-Strogatz networks.")
    model.add_argument("--watts-strogatz-p", type=float, default=d.watts_strogatz_p,
                       help="Rewiring probability for Watts-Strogatz networks.")

    regional = p.add_argument_group("regional simulation parameters")
    regional.add_argument("--config", metavar="PATH", default=None,
                          help="Load the entire configuration from a JSON file "
                               "(overrides all other options). Best way to set "
                               "heterogeneous city_populations, an asymmetric "
                               "travel_matrix, and trip_duration_distribution.")
    regional.add_argument("--number-of-cities", type=int, default=d.number_of_cities,
                          help="Number of cities (when --city-populations unset).")
    regional.add_argument("--population-per-city", type=int,
                          default=d.population_per_city,
                          help="Population per city (when --city-populations unset).")
    regional.add_argument("--city-populations", default=None,
                          help="Comma-separated per-city sizes, e.g. '500,200,1500'. "
                               "Overrides --number-of-cities/--population-per-city.")
    regional.add_argument("--travel-fraction", type=float, default=d.travel_fraction,
                          help="Fraction of population eligible to travel (0-0.5).")
    regional.add_argument("--daily-travel-rate", type=float,
                          default=d.daily_travel_rate,
                          help="Per-eligible daily travel probability (used to "
                               "build the default uniform travel matrix).")

    exp = p.add_argument_group("experiments")
    exp.add_argument("--experiment", type=int, default=0, metavar="N",
                     help="Run the regional config N times over different seeds "
                          "and report mean/std (0 = single run).")
    exp.add_argument("--experiment-base-seed", type=int, default=0,
                     help="First seed used by --experiment.")
    exp.add_argument("--sensitivity-config", metavar="PATH", default=None,
                     help="Run a sensitivity sweep: a JSON file mapping Config "
                          "field names to a list of values to try, e.g. "
                          '\'{"daily_travel_rate": [0.0, 0.1, 0.2], '
                          '"number_of_cities": [2, 5, 10]}\'. Sweeps the full '
                          "Cartesian product against --config/CLI as the base.")
    exp.add_argument("--sensitivity-runs-per-combo", type=int, default=5,
                     help="Independent seeds run per --sensitivity-config grid point.")
    exp.add_argument("--sensitivity-csv", metavar="PATH", default="sensitivity_results.csv",
                     help="Where to write every sensitivity-sweep run as a CSV row.")

    out = p.add_argument_group("visualization and output")
    out.add_argument("--visualization-mode",
                     choices=("auto", "network", "cluster", "heatmap", "pie"),
                     default=d.visualization_mode,
                     help="Regional animation mode. 'auto' (default) picks "
                          "network/cluster/heatmap from the largest city's "
                          "population.")
    out.add_argument("--heatmap-tile-percent", type=float,
                     default=100 * d.heatmap_tile_fraction, metavar="PERCENT",
                     help="Approximate population represented by one heatmap "
                          "tile (5 creates 20 tiles per city).")
    out.add_argument("--layout", choices=("grid", "circle"), default="grid",
                     help="Fixed arrangement of individuals in the animation.")
    out.add_argument("--interval-ms", type=int, default=400,
                     help="Milliseconds per animation frame.")
    out.add_argument("--save-gif", metavar="PATH", default=None,
                     help="Save the state animation to a .gif file.")
    out.add_argument("--save-curves", metavar="PATH", default=None,
                     help="Save the SEIR curves to an image file (.png).")
    out.add_argument("--export-csv", metavar="PATH", default=None,
                     help="Export the day-by-day history to a CSV file.")
    out.add_argument("--show", action="store_true",
                     help="Open interactive windows for the outputs.")
    out.add_argument("--quiet", action="store_true",
                     help="Suppress per-day progress output.")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    """Translate parsed CLI arguments into a validated :class:`Config`.

    If ``--config`` is given, the JSON file is loaded and used as-is (the
    "change only the configuration" entry point for heterogeneous city
    populations, asymmetric travel matrices, and trip-duration distributions);
    all other CLI flags are ignored in that case. Otherwise every field is
    built from individual CLI flags, with ``--city-populations`` (a
    comma-separated list) overriding ``--number-of-cities``/
    ``--population-per-city`` when given.

    Args:
        args: The namespace returned by the argument parser.

    Returns:
        A validated configuration (raises ``ValueError`` on bad values).
    """
    if args.config:
        return Config.from_json(args.config)

    city_populations = None
    if args.city_populations:
        city_populations = tuple(
            int(p.strip()) for p in args.city_populations.split(",") if p.strip())

    return Config(
        population_size=args.population_size,
        daily_contacts=args.daily_contacts,
        infection_probability=args.infection_probability,
        incubation_days=args.incubation_days,
        infectious_days=args.infectious_days,
        initial_infected=args.initial_infected,
        simulation_days=args.simulation_days,
        random_seed=args.random_seed,
        contact_model=args.contact_model,
        random_degree_min=args.random_degree_min,
        random_degree_max=args.random_degree_max,
        watts_strogatz_k=args.watts_strogatz_k,
        watts_strogatz_p=args.watts_strogatz_p,
        number_of_cities=args.number_of_cities,
        population_per_city=args.population_per_city,
        city_populations=city_populations,
        travel_fraction=args.travel_fraction,
        daily_travel_rate=args.daily_travel_rate,
        visualization_mode=args.visualization_mode,
        heatmap_tile_fraction=args.heatmap_tile_percent / 100,
    )


def print_report(simulation: Simulation, config: Config) -> None:
    """Print the end-of-run epidemic summary for a single city.

    Args:
        simulation: The completed simulation.
        config: The run configuration.
    """
    stats = summary(simulation.history)
    print("\n" + "=" * 48)
    print("  EPIDEMIC SUMMARY (single city)")
    print("=" * 48)
    print(f"  Population:          {int(stats['population'])}")
    print(f"  Estimated R0:        {config.estimated_r0():.2f}")
    print(f"  Peak infectious:     {int(stats['peak_infectious'])} "
          f"(day {int(stats['peak_infectious_day'])})")
    print(f"  Peak exposed:        {int(stats['peak_exposed'])} "
          f"(day {int(stats['peak_exposed_day'])})")
    print(f"  Total ever infected: {int(stats['total_infected'])}")
    print(f"  Attack rate:         {100 * stats['attack_rate']:.1f}%")
    print(f"  Epidemic duration:   {int(stats['epidemic_duration_days'])} days")
    print(f"  Final susceptible:   {int(stats['final_susceptible'])}")
    print("=" * 48 + "\n")


def print_regional_report(regional_sim: RegionalSimulation,
                          config: Config) -> None:
    """Print the end-of-run summary for a regional simulation.

    Args:
        regional_sim: The completed RegionalSimulation.
        config: The run configuration.
    """
    summary_data = regional_sim.regional_summary()
    sources = summary_data["infection_sources"]

    print("\n" + "=" * 60)
    print("  REGIONAL EPIDEMIC SUMMARY")
    print("=" * 60)
    print(f"  Number of cities:      {summary_data['num_cities']}")
    print(f"  City populations:      {config.city_sizes()}")
    print(f"  Total population:      {int(summary_data['total_population'])}")
    print(f"  Total regional infections: {int(summary_data['total_regional_infections'])}")
    print(f"  Regional attack rate:  {100 * summary_data['regional_attack_rate']:.1f}%")
    print(f"  Cities reached:        {summary_data['cities_reached']}/{summary_data['num_cities']}")
    if summary_data["average_arrival_delay"] >= 0:
        print(f"  Avg arrival delay:     {summary_data['average_arrival_delay']:.1f} days")
    print(f"  Estimated R0:          {config.estimated_r0():.2f}")
    print()

    for city, city_stats in zip(regional_sim.cities, summary_data["city_summaries"]):
        first_day = city_stats["first_infection_day"]
        first_txt = f"day {int(first_day)}" if first_day >= 0 else "never"
        seeded = " (seeded)" if city.id == 0 else ""
        source = sources[city.id]
        source_txt = (" (seed city)" if city.id == 0 else
                     f" (source: City {city_label(source)})" if source >= 0 else
                     " (source: none)")
        print(f"  City {city_label(city.id)}{seeded} "
              f"[n={int(city_stats['population'])}]{source_txt}:")
        print(f"    First infection:   {first_txt}")
        print(f"    Peak infectious:   {int(city_stats['peak_infectious'])} "
              f"(day {int(city_stats['peak_infectious_day'])})")
        print(f"    Peak exposed:      {int(city_stats['peak_exposed'])} "
              f"(day {int(city_stats['peak_exposed_day'])})")
        print(f"    Peak recovered:    {int(city_stats['peak_recovered'])} "
              f"(day {int(city_stats['peak_recovered_day'])})")
        print(f"    Attack rate:       {100 * city_stats['attack_rate']:.1f}%")
        print(f"    Epidemic duration: {int(city_stats['epidemic_duration_days'])} days")
        print(f"    Imported cases:    {int(city_stats['imported_infections'])}")
        print(f"    Exported cases:    {int(city_stats['exported_infections'])}")

    print()
    print("  Travel statistics:")
    print(f"    Total travel events (completed trips): {summary_data['num_travel_events']}")
    print(f"    Imported infections:                   {summary_data['imported_infections']}")
    if summary_data["num_cities"] > 1:
        b_day = summary_data.get("city_b_first_infection_day", -1.0)
        b_txt = f"day {int(b_day)}" if b_day >= 0 else "never"
        print(f"    Infection reached City B:               {b_txt}")
    print("=" * 60 + "\n")


def main(argv: Optional[List[str]] = None) -> None:
    """Parse arguments, run the simulation, and produce requested outputs.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``); exposed for
            testing.
    """
    args = build_parser().parse_args(argv)
    config = config_from_args(args)

    if args.sensitivity_config:
        run_sensitivity_mode(config, args)
    elif args.experiment > 0:
        run_experiment_mode(config, args)
    elif args.single_city:
        run_single_city(config, args)
    else:
        run_regional(config, args)


def run_single_city(config: Config, args: argparse.Namespace) -> None:
    """Run a single-city SEIR simulation.

    Args:
        config: The validated configuration.
        args: The parsed command-line arguments.
    """
    print("Running single-city SEIR simulation...")
    print(f"  Contact model: {config.contact_model}")
    print(f"  Population: {config.population_size}")
    print(f"  Seed: {config.random_seed}")

    simulation = Simulation(config)
    simulation.run(verbose=not args.quiet)
    print_report(simulation, config)

    if args.export_csv:
        export_csv(simulation.history, args.export_csv)
        print(f"Exported history to {args.export_csv}")

    if args.save_curves or args.show:
        visualization.plot_curves(
            simulation.history, config,
            save_path=args.save_curves, show=args.show)
    if args.save_gif or args.show:
        visualization.animate_states(
            simulation.state_frames, simulation.history, config,
            layout=args.layout, save_path=args.save_gif,
            show=args.show, interval_ms=args.interval_ms,
            graph=getattr(simulation.engine.contact_model, "graph", None))


def run_regional(config: Config, args: argparse.Namespace) -> None:
    """Run a regional multi-city SEIR simulation.

    Args:
        config: The validated configuration.
        args: The parsed command-line arguments.
    """
    print("Running regional multi-city SEIR simulation...")
    sizes = config.city_sizes()
    print(f"  Cities: {config.num_cities()}   Populations: {sizes}")
    print(f"  Contact model: {config.contact_model}")
    print(f"  Travel fraction: {config.travel_fraction}")
    print(f"  Daily travel rate: {config.daily_travel_rate}")
    print(f"  Visualization mode: {config.visualization_mode}")
    if config.visualization_mode == "heatmap":
        print(f"  Heatmap tile size: {100 * config.heatmap_tile_fraction:g}% of each city")
    print(f"  Seed: {config.random_seed}")

    regional_sim = RegionalSimulation(config)
    regional_sim.run(verbose=not args.quiet)
    print_regional_report(regional_sim, config)

    if args.save_curves or args.show:
        visualization.plot_regional_curves(
            regional_sim, save_path=args.save_curves, show=args.show)
    if args.save_gif or args.show:
        visualization.animate_regional(
            regional_sim, mode=config.visualization_mode, layout=args.layout,
            save_path=args.save_gif, show=args.show, interval_ms=args.interval_ms,
            heatmap_tile_fraction=config.heatmap_tile_fraction)


def run_experiment_mode(config: Config, args: argparse.Namespace) -> None:
    """Run a repeated-simulation experiment and aggregate the statistics.

    Args:
        config: The validated configuration (used as the template).
        args: The parsed command-line arguments.
    """
    print(f"Running experiment: {args.experiment} simulations (seeds "
          f"{args.experiment_base_seed}..{args.experiment_base_seed + args.experiment - 1})...")
    result = run_experiment(config, num_runs=args.experiment,
                            base_seed=args.experiment_base_seed,
                            verbose=not args.quiet)
    print_experiment_report(result, config)


def run_sensitivity_mode(config: Config, args: argparse.Namespace) -> None:
    """Run a sensitivity-analysis parameter sweep and write results to CSV.

    Args:
        config: The validated base configuration (grid values override it).
        args: The parsed command-line arguments.
    """
    with open(args.sensitivity_config, "r", encoding="utf-8") as handle:
        param_grid = json.load(handle)

    combos = 1
    for values in param_grid.values():
        combos *= len(values)
    total_runs = combos * args.sensitivity_runs_per_combo
    print(f"Running sensitivity analysis: {combos} parameter combinations x "
          f"{args.sensitivity_runs_per_combo} seeds = {total_runs} simulations...")
    print(f"  Parameter grid: {param_grid}")

    run_sensitivity_analysis(
        config, param_grid,
        runs_per_combo=args.sensitivity_runs_per_combo,
        csv_path=args.sensitivity_csv,
        verbose=not args.quiet)


if __name__ == "__main__":
    main()
