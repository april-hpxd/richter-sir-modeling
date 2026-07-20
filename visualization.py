"""Visualization: a fixed-position scatter animation and SEIR curves.

This module is purely presentational. It consumes recorded simulation output
(per-day state snapshots and :class:`~simulation.DailyRecord` counts) and knows
nothing about disease dynamics -- its only tie to the model is a colour lookup
keyed by :class:`~disease_model.State`.

Products:

* :func:`animate_states` -- an animated GIF in which each of the (small)
  population is drawn as one marker at a **fixed** position (grid or circle).
  Only the marker colours change from day to day, so the eye tracks individuals
  turning exposed, infectious, then recovered.
* :func:`plot_curves` -- the standard SEIR epidemic curves over time.
* :func:`animate_regional_states` -- side-by-side animations of multiple cities.
* :func:`plot_regional_curves` -- regional epidemic curves (aggregated).

Colour convention: Susceptible = light grey, Exposed = orange,
Infectious = red, Recovered = green.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, Patch

from config import Config
from disease_model import State
from regional_simulation import RegionalSimulation
from simulation import DailyRecord

# --- Shared colour scheme ----------------------------------------------------
STATE_COLOR: Dict[State, str] = {
    State.SUSCEPTIBLE: "#d9d9d9",  # light grey
    State.EXPOSED: "#f4a261",      # orange
    State.INFECTIOUS: "#e63946",   # red
    State.RECOVERED: "#2a9d8f",    # green
}
STATE_LABEL: Dict[State, str] = {
    State.SUSCEPTIBLE: "Susceptible",
    State.EXPOSED: "Exposed",
    State.INFECTIOUS: "Infectious",
    State.RECOVERED: "Recovered",
}

Position = Tuple[float, float]


def city_label(city_id: int) -> str:
    """Return a human-friendly label for a city ("A", "B", ..., then "City 26").

    Matches the spec's "City A / City B" naming for the first 26 cities and
    falls back gracefully for larger regions.

    Args:
        city_id: Zero-based city index.

    Returns:
        A short label string.
    """
    return chr(ord("A") + city_id) if 0 <= city_id < 26 else f"City {city_id}"


def grid_layout(n: int, cols: int = 10) -> List[Position]:
    """Arrange ``n`` individuals on a fixed rectangular grid.

    Individuals are placed left-to-right, top-to-bottom, so person ``i`` always
    occupies the same cell. With the default 50 people and 10 columns this is
    the 5x10 grid suggested for the milestone.

    Args:
        n: Number of individuals.
        cols: Number of columns in the grid.

    Returns:
        A list of ``(x, y)`` positions, indexed by individual id.
    """
    positions: List[Position] = []
    for i in range(n):
        row, col = divmod(i, cols)
        positions.append((float(col), float(-row)))  # negative y: top-down
    return positions


def circle_layout(n: int, radius: float = 1.0) -> List[Position]:
    """Arrange ``n`` individuals evenly around a fixed circle.

    Args:
        n: Number of individuals.
        radius: Circle radius.

    Returns:
        A list of ``(x, y)`` positions, indexed by individual id.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(radius * float(np.cos(a)), radius * float(np.sin(a)))
            for a in angles]


def _legend_handles() -> List[Patch]:
    """Return legend patches for the S/E/I/R colour scheme."""
    return [Patch(color=STATE_COLOR[s], label=STATE_LABEL[s]) for s in State]


def animate_states(state_frames: List[List[State]], history: List[DailyRecord],
                   config: Config, layout: str = "grid",
                   save_path: Optional[str] = None,
                   show: bool = False,
                   interval_ms: int = 400) -> FuncAnimation:
    """Animate per-individual state changes on a fixed layout.

    Args:
        state_frames: Per-day list of per-individual states (from
            :attr:`~simulation.Simulation.state_frames`).
        history: Matching per-day :class:`DailyRecord` counts, used for the
            on-screen tally.
        config: The run configuration (for the population size / titling).
        layout: ``"grid"`` (default) or ``"circle"`` positioning.
        save_path: If given, save the animation to this ``.gif`` path.
        show: If ``True``, display the animation window.
        interval_ms: Delay between frames in milliseconds.

    Returns:
        The :class:`~matplotlib.animation.FuncAnimation`. Keep a reference to it
        alive until display/saving completes.
    """
    n = config.population_size
    positions = (circle_layout(n) if layout == "circle"
                 else grid_layout(n))
    coords = np.array(positions)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("SEIR epidemic in a well-mixed city of "
                 f"{n} people", fontsize=13, weight="bold")

    marker_size = max(80, min(600, 12000 / n))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], s=marker_size,
        c=[STATE_COLOR[s] for s in state_frames[0]],
        edgecolors="#555555", linewidths=0.5, zorder=2)

    ax.set_aspect("equal")
    ax.axis("off")
    pad = 1.0
    ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)
    ax.legend(handles=_legend_handles(), loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False)

    day_text = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top",
                       ha="left", family="monospace", fontsize=10)

    def update(frame: int):
        """Recolour every marker to its state on day ``frame``."""
        states = state_frames[frame]
        scatter.set_color([STATE_COLOR[s] for s in states])
        # set_color drops the edge styling; restore it for crisp markers.
        scatter.set_edgecolors("#555555")
        rec = history[frame]
        day_text.set_text(
            f"Day {rec.day}\n"
            f"S={rec.susceptible}  E={rec.exposed}  "
            f"I={rec.infectious}  R={rec.recovered}")
        return scatter, day_text

    anim = FuncAnimation(fig, update, frames=len(state_frames),
                         interval=interval_ms, blit=False, repeat=False)

    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    if save_path:
        fps = max(1, round(1000 / interval_ms))
        anim.save(save_path, writer=PillowWriter(fps=fps))
        print(f"Saved animation to {save_path}")
    if show:
        plt.show()
    return anim


def plot_curves(history: List[DailyRecord], config: Config,
                save_path: Optional[str] = None,
                show: bool = False) -> plt.Figure:
    """Plot the standard SEIR epidemic curves over time.

    Args:
        history: The recorded simulation history.
        config: The run configuration (for the population reference line).
        save_path: If given, save the figure to this image path (e.g. ``.png``).
        show: If ``True``, display the figure.

    Returns:
        The created :class:`matplotlib.figure.Figure`.
    """
    days = [r.day for r in history]
    series = {
        State.SUSCEPTIBLE: [r.susceptible for r in history],
        State.EXPOSED: [r.exposed for r in history],
        State.INFECTIOUS: [r.infectious for r in history],
        State.RECOVERED: [r.recovered for r in history],
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for state, values in series.items():
        ax.plot(days, values, color=STATE_COLOR[state],
                label=STATE_LABEL[state], linewidth=2.2)

    ax.set_title("SEIR epidemic curves", fontsize=13, weight="bold")
    ax.set_xlabel("Day")
    ax.set_ylabel("Individuals")
    ax.set_ylim(0, config.population_size)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"Saved SEIR curves to {save_path}")
    if show:
        plt.show()
    return fig


def animate_regional_states(regional_sim: RegionalSimulation,
                            layout: str = "grid",
                            save_path: Optional[str] = None,
                            show: bool = False,
                            interval_ms: int = 400,
                            show_travel: bool = True) -> FuncAnimation:
    """Animate multiple cities side-by-side, with optional travel arrows.

    All cities animate simultaneously on a shared timeline. When
    ``show_travel`` is set, a dashed arrow is drawn between neighbouring city
    panels on each day that has travel between them, labelled with the number
    of travelers that day (both directions combined).

    Args:
        regional_sim: The completed RegionalSimulation with multiple cities.
        layout: ``"grid"`` (default) or ``"circle"`` positioning.
        save_path: If given, save the animation to this ``.gif`` path.
        show: If ``True``, display the animation window.
        interval_ms: Delay between frames in milliseconds.
        show_travel: If ``True``, draw dashed travel arrows between adjacent
            city panels for each day.

    Returns:
        The :class:`~matplotlib.animation.FuncAnimation`.
    """
    cities = regional_sim.cities
    num_cities = len(cities)
    pop_per_city = regional_sim.config.population_per_city

    # Per-day travel counts between each ordered city pair, for the arrows.
    travel_by_day: Dict[int, Dict[Tuple[int, int], int]] = defaultdict(
        lambda: defaultdict(int))
    for te in regional_sim.travel_events:
        travel_by_day[te.day][(te.home_city_id, te.destination_city_id)] += 1

    positions_per_city = [
        (circle_layout(pop_per_city) if layout == "circle"
         else grid_layout(pop_per_city))
        for _ in cities
    ]

    fig, axes = plt.subplots(1, num_cities, figsize=(6*num_cities, 5))
    if num_cities == 1:
        axes = [axes]

    fig.suptitle(
        f"Regional SEIR ({num_cities} cities, "
        f"{pop_per_city} people each)",
        fontsize=13, weight="bold"
    )

    marker_size = max(80, min(600, 12000 / pop_per_city))
    scatters = []
    day_texts = []

    for city_idx, (ax, city, positions) in enumerate(
        zip(axes, cities, positions_per_city)
    ):
        coords = np.array(positions)
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], s=marker_size,
            c=[STATE_COLOR[s] for s in cities[city_idx].state_frames[0]],
            edgecolors="#555555", linewidths=0.5, zorder=2
        )
        scatters.append(scatter)

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"City {city_label(city_idx)}", fontsize=11, weight="bold")
        pad = 1.0
        ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
        ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)

        day_text = ax.text(
            0.01, 0.99, "", transform=ax.transAxes, va="top",
            ha="left", family="monospace", fontsize=9
        )
        day_texts.append(day_text)

    # Add legend once, outside the subplots
    fig.legend(handles=_legend_handles(), loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False)

    # Transient travel artists (arrows + labels), recreated each frame.
    travel_artists: List = []

    def _draw_travel(frame: int) -> None:
        """Draw dashed arrows between adjacent panels for day ``frame``."""
        while travel_artists:
            travel_artists.pop().remove()
        if not show_travel or num_cities < 2:
            return
        day_travel = travel_by_day.get(frame, {})
        for i in range(num_cities - 1):
            count = day_travel.get((i, i + 1), 0) + day_travel.get((i + 1, i), 0)
            if count == 0:
                continue
            # Midpoint band between the two adjacent axes, in figure fraction.
            left = axes[i].get_position()
            right = axes[i + 1].get_position()
            x0 = left.x1 - 0.01
            x1 = right.x0 + 0.01
            y = (left.y0 + left.y1) / 2
            arrow = FancyArrowPatch(
                (x0, y), (x1, y), transform=fig.transFigure,
                arrowstyle="<|-|>", mutation_scale=14, linewidth=1.4,
                linestyle=(0, (4, 3)), color="#4062bb", zorder=5)
            fig.add_artist(arrow)
            label = fig.text((x0 + x1) / 2, y + 0.05, f"{count} trav.",
                             ha="center", va="bottom", fontsize=8,
                             color="#4062bb", weight="bold")
            travel_artists.extend([arrow, label])

    def update(frame: int):
        """Recolour every marker to its state on day ``frame``."""
        for city_idx, city in enumerate(cities):
            if frame < len(city.state_frames):
                states = city.state_frames[frame]
                scatters[city_idx].set_color([STATE_COLOR[s] for s in states])
                scatters[city_idx].set_edgecolors("#555555")

                if frame < len(city.history):
                    rec = city.history[frame]
                    day_texts[city_idx].set_text(
                        f"Day {rec.day}\n"
                        f"S={rec.susceptible} E={rec.exposed}\n"
                        f"I={rec.infectious} R={rec.recovered}"
                    )
        _draw_travel(frame)
        return scatters + day_texts + travel_artists

    max_frames = max(len(city.state_frames) for city in cities)
    anim = FuncAnimation(
        fig, update, frames=max_frames,
        interval=interval_ms, blit=False, repeat=False
    )

    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    if save_path:
        fps = max(1, round(1000 / interval_ms))
        anim.save(save_path, writer=PillowWriter(fps=fps))
        print(f"Saved regional animation to {save_path}")
    if show:
        plt.show()
    return anim


def plot_regional_curves(regional_sim: RegionalSimulation,
                         save_path: Optional[str] = None,
                         show: bool = False) -> plt.Figure:
    """Plot regional aggregated SEIR curves, one subplot per city.

    Args:
        regional_sim: The completed RegionalSimulation.
        save_path: If given, save the figure to this image path.
        show: If ``True``, display the figure.

    Returns:
        The created :class:`matplotlib.figure.Figure`.
    """
    cities = regional_sim.cities
    num_cities = len(cities)
    pop_per_city = regional_sim.config.population_per_city

    fig, axes = plt.subplots(1, num_cities, figsize=(7*num_cities, 5))
    if num_cities == 1:
        axes = [axes]

    fig.suptitle(
        f"Regional SEIR Curves ({num_cities} cities)",
        fontsize=13, weight="bold"
    )

    for city_idx, (ax, city) in enumerate(zip(axes, cities)):
        history = city.history
        if not history:
            continue

        days = [r.day for r in history]
        series = {
            State.SUSCEPTIBLE: [r.susceptible for r in history],
            State.EXPOSED: [r.exposed for r in history],
            State.INFECTIOUS: [r.infectious for r in history],
            State.RECOVERED: [r.recovered for r in history],
        }

        for state, values in series.items():
            ax.plot(days, values, color=STATE_COLOR[state],
                    label=STATE_LABEL[state], linewidth=2.0)

        ax.set_title(f"City {city_label(city_idx)}", fontsize=11, weight="bold")
        ax.set_xlabel("Day")
        ax.set_ylabel("Individuals")
        ax.set_ylim(0, pop_per_city)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"Saved regional curves to {save_path}")
    if show:
        plt.show()
    return fig
