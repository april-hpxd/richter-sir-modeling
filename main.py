"""Command-line entry point for the SEIR well-mixed city simulation.

Thin wiring layer: it parses command-line options into a
:class:`~config.Config`, runs one well-mixed SEIR epidemic, prints a summary,
and produces the requested outputs (animated GIF, SEIR curves, CSV). All real
work lives in the dedicated modules.

Examples
--------
Run with defaults, save the animation and curves::

    python main.py --save-gif epidemic.gif --save-curves curves.png

A more transmissible outbreak, arranged on a circle, exported to CSV::

    python main.py --infection-probability 0.08 --layout circle \\
        --export-csv run.csv
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from config import Config
from simulation import Simulation
from statistics import export_csv, summary
import visualization


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser.

    Every :class:`~config.Config` field is exposed as an option defaulting to
    the value in :class:`Config`, so ``python main.py`` alone is valid.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    d = Config()
    p = argparse.ArgumentParser(
        description="Well-mixed SEIR epidemic simulation "
                    "(disease-engine validation milestone).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    model = p.add_argument_group("model parameters")
    model.add_argument("--population-size", type=int, default=d.population_size,
                       help="Number of individuals in the city.")
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
    )


def print_report(simulation: Simulation, config: Config) -> None:
    """Print the end-of-run epidemic summary.

    Args:
        simulation: The completed simulation.
        config: The run configuration.
    """
    stats = summary(simulation.history)
    print("\n" + "=" * 48)
    print("  EPIDEMIC SUMMARY (well-mixed SEIR)")
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


def main(argv: Optional[List[str]] = None) -> None:
    """Parse arguments, run the simulation, and produce requested outputs.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``); exposed for
            testing.
    """
    args = build_parser().parse_args(argv)
    config = config_from_args(args)

    print("Running well-mixed SEIR simulation...")
    print(f"  Parameters: {config.as_dict()}")

    simulation = Simulation(config)
    simulation.run(verbose=not args.quiet)
    print_report(simulation, config)

    if args.export_csv:
        export_csv(simulation.history, args.export_csv)
        print(f"Exported history to {args.export_csv}")

    # Static curves first, then the (optionally blocking) animation.
    visualization.plot_curves(
        simulation.history, config,
        save_path=args.save_curves, show=args.show)
    visualization.animate_states(
        simulation.state_frames, simulation.history, config,
        layout=args.layout, save_path=args.save_gif,
        show=args.show, interval_ms=args.interval_ms)


if __name__ == "__main__":
    main()
