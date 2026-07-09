"""Command-line entry point for the single-city network SIR simulation.

This is the thin "wiring" layer for Milestone 1. It parses command-line
options into a :class:`~config.Config`, runs one city's epidemic, prints a
summary, and launches the requested visualizations. All the real work lives in
the dedicated modules; ``main`` only orchestrates them.

"""

from __future__ import annotations

if __name__ == "__main__":
    main()
