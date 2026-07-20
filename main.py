"""Command-line entry point for the SEIR regional multi-city simulation.

Thin wiring layer: it parses command-line options into a
:class:`~config.Config`, runs either a single-city or regional multi-city SEIR
epidemic, prints a summary, and produces the requested outputs (animated GIF,
SEIR curves, CSV). All real work lives in the dedicated modules.

Examples
--------
Single city with defaults::

    python main.py --single-city --save-gif epidemic.gif --save-curves curves.png

Regional simulation (2 cities)::

    python main.py --regional --number-of-cities 2 --save-gif regional.gif

Two cities with travel::

    python main.py --regional --number-of-cities 2 --travel-fraction 0.5 \\
        --daily-travel-rate 0.1 --save-gif travel.gif
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from config import Config
from regional_simulation import RegionalSimulation
from simulation import Simulation
from statistics import export_csv, summary
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
    model.add_argument("--contact-model", choices=("well-mixed", "watts-strogatz"),
                       default=d.contact_model,
                       help="Contact network model.")
    model.add_argument("--watts-strogatz-k", type=int, default=d.watts_strogatz_k,
                       help="Neighbourhood size for Watts-Strogatz networks.")
    model.add_argument("--watts-strogatz-p", type=float, default=d.watts_strogatz_p,
                       help="Rewiring probability for Watts-Strogatz networks.")

    regional = p.add_argument_group("regional simulation parameters")
    regional.add_argument("--number-of-cities", type=int, default=d.number_of_cities,
                          help="Number of cities in the regional simulation.")
    regional.add_argument("--population-per-city", type=int,
                          default=d.population_per_city,
                          help="Number of individuals per city.")
    regional.add_argument("--travel-fraction", type=float, default=d.travel_fraction,
                          help="Fraction of population eligible to travel (0-1).")
    regional.add_argument("--daily-travel-rate", type=float,
                          default=d.daily_travel_rate,
                          help="Fraction of eligible travelers who actually travel.")

    out = p.add_argument_group("visualization and output")
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

    Args:
        args: The namespace returned by the argument parser.

    Returns:
        A validated configuration (raises ``ValueError`` on bad values).
    """
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
        watts_strogatz_k=args.watts_strogatz_k,
        watts_strogatz_p=args.watts_strogatz_p,
        number_of_cities=args.number_of_cities,
        population_per_city=args.population_per_city,
        travel_fraction=args.travel_fraction,
        daily_travel_rate=args.daily_travel_rate,
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

    print("\n" + "=" * 60)
    print("  REGIONAL EPIDEMIC SUMMARY")
    print("=" * 60)
    print(f"  Number of cities:    {summary_data['num_cities']}")
    print(f"  Total population:    {int(summary_data['total_population'])}")
    print(f"  Total ever infected: {int(summary_data['total_infected'])}")
    print(f"  Regional attack rate: {100 * summary_data['regional_attack_rate']:.1f}%")
    print(f"  Estimated R0:        {config.estimated_r0():.2f}")
    print()

    for city in regional_sim.cities:
        city_stats = city.summary_stats()
        first_day = city_stats["first_infection_day"]
        first_txt = f"day {int(first_day)}" if first_day >= 0 else "never"
        seeded = " (seeded)" if city.id == 0 else ""
        print(f"  City {city_label(city.id)}{seeded}:")
        print(f"    First infection:   {first_txt}")
        print(f"    Peak infectious:   {int(city_stats['peak_infectious'])} "
              f"(day {int(city_stats['peak_infectious_day'])})")
        print(f"    Peak exposed:      {int(city_stats['peak_exposed'])} "
              f"(day {int(city_stats['peak_exposed_day'])})")
        print(f"    Attack rate:       {100 * city_stats['attack_rate']:.1f}%")
        print(f"    Epidemic duration: {int(city_stats['epidemic_duration_days'])} days")
        print(f"    Imported cases:    {int(city_stats['imported_infections'])}")

    b_day = summary_data.get("city_b_first_infection_day", -1.0)
    print()
    print("  Travel statistics:")
    print(f"    Total travel events:      {summary_data['num_travel_events']}")
    print(f"    Imported infections:      {summary_data['imported_infections']}")
    if summary_data["num_cities"] > 1:
        b_txt = f"day {int(b_day)}" if b_day >= 0 else "never"
        print(f"    Infection reached City B: {b_txt}")
    print("=" * 60 + "\n")


def main(argv: Optional[List[str]] = None) -> None:
    """Parse arguments, run the simulation, and produce requested outputs.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``); exposed for
            testing.
    """
    args = build_parser().parse_args(argv)
    config = config_from_args(args)

    if args.single_city:
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
            show=args.show, interval_ms=args.interval_ms)


def run_regional(config: Config, args: argparse.Namespace) -> None:
    """Run a regional multi-city SEIR simulation.

    Args:
        config: The validated configuration.
        args: The parsed command-line arguments.
    """
    print("Running regional multi-city SEIR simulation...")
    print(f"  Number of cities: {config.number_of_cities}")
    print(f"  Population per city: {config.population_per_city}")
    print(f"  Contact model: {config.contact_model}")
    print(f"  Travel fraction: {config.travel_fraction}")
    print(f"  Daily travel rate: {config.daily_travel_rate}")
    print(f"  Seed: {config.random_seed}")

    # RegionalSimulation seeds City A only and records the day-0 baseline for
    # every city (City B and beyond start 100% susceptible).
    regional_sim = RegionalSimulation(config)
    regional_sim.run(verbose=not args.quiet)
    print_regional_report(regional_sim, config)

    if args.save_curves or args.show:
        visualization.plot_regional_curves(
            regional_sim, save_path=args.save_curves, show=args.show)
    if args.save_gif or args.show:
        visualization.animate_regional_states(
            regional_sim, layout=args.layout, save_path=args.save_gif,
            show=args.show, interval_ms=args.interval_ms)


if __name__ == "__main__":
    main()
