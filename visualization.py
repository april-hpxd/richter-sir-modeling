"""Visualization: contact-network animations and SEIR curves.

Colour convention: Susceptible = light grey, Exposed = orange,
Infectious = red, Recovered = green
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch, Patch

from config import CLUSTER_MAX_POPULATION, Config, NETWORK_MAX_POPULATION
from disease_model import State
from regional_simulation import RegionalSimulation
from simulation import DailyRecord

#  Shared colour scheme 
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


def cluster_layout(n: int, cluster_of: np.ndarray) -> List[Position]:
    """Arrange individuals so cluster-mates sit close together on-screen.

    Clusters are placed on a coarse grid (their own local region of the
    figure); within a cluster, members are packed on a small sub-grid around
    that region's center. This makes the :class:`~interaction.ClusteredContactModel`
    structure visually obvious -- dense local neighbourhoods with only a few
    edges reaching across to other regions -- and scales automatically with
    both population size and the number of clusters (no layout is hardcoded
    to a particular cluster count).

    Args:
        n: Number of individuals.
        cluster_of: Per-individual cluster id (from the contact model).

    Returns:
        A list of ``(x, y)`` positions, indexed by individual id.
    """
    cluster_ids = sorted(set(int(c) for c in cluster_of))
    cols = max(1, int(np.ceil(np.sqrt(len(cluster_ids)))))
    spacing = 3.5
    positions: List[Optional[Position]] = [None] * n
    for idx, cid in enumerate(cluster_ids):
        row, col = divmod(idx, cols)
        cx, cy = col * spacing, -row * spacing
        members = [i for i in range(n) if int(cluster_of[i]) == cid]
        member_cols = max(1, int(np.ceil(np.sqrt(len(members)))))
        for j, node_id in enumerate(members):
            mrow, mcol = divmod(j, member_cols)
            positions[node_id] = (cx + mcol * 0.55, cy - mrow * 0.55)
    return positions


def _resolve_layout(n: int, layout: str,
                    cluster_of: Optional[np.ndarray]) -> List[Position]:
    """Pick node positions: cluster-aware when a cluster assignment exists,
    otherwise the requested fixed grid/circle layout."""
    if cluster_of is not None:
        return cluster_layout(n, cluster_of)
    return circle_layout(n) if layout == "circle" else grid_layout(n)


def _legend_handles() -> List[Patch]:
    """Return legend patches for the S/E/I/R colour scheme."""
    return [Patch(color=STATE_COLOR[s], label=STATE_LABEL[s]) for s in State]


def _draw_network_edges(ax, graph, coords: np.ndarray) -> None:
    """Draw the static, within-city contact graph beneath the node markers."""
    if graph is None or graph.number_of_edges() == 0:
        return
    segments = [(coords[left], coords[right]) for left, right in graph.edges()]
    ax.add_collection(LineCollection(
        segments, colors="#9aa0a6", linewidths=0.65, alpha=0.55, zorder=1
    ))


#
# Node highlight helpers: newly infected, fading recovered, traveler rings,
# and brief transmission flashes. Positions never change -- only colour,
# marker edge, and short-lived overlay artists do.
#
RECOVERED_FADE_DAYS = 15
RECOVERED_MIN_ALPHA = 0.35
NEWLY_INFECTED_RING_COLOR = "#ffd60a"
TRAVELER_RING_COLOR = "#4062bb"
TRANSMISSION_FADE_FRAMES = 3


def _recovered_run_lengths(state_frames: List[List[State]]) -> np.ndarray:
    """Return, for every ``(frame, node)``, consecutive days spent RECOVERED.

    A pure function over the existing per-day state snapshots -- no new data
    is recorded to support the recovered-node fade effect.
    """
    is_recovered = np.array(
        [[s is State.RECOVERED for s in frame] for frame in state_frames])
    lengths = np.zeros_like(is_recovered, dtype=int)
    for f in range(len(state_frames)):
        if f == 0:
            lengths[f] = is_recovered[f].astype(int)
        else:
            lengths[f] = np.where(is_recovered[f], lengths[f - 1] + 1, 0)
    return lengths


def _newly_infected_mask(state_frames: List[List[State]], frame: int) -> np.ndarray:
    """Boolean mask of nodes that just became EXPOSED on ``frame``."""
    current = state_frames[frame]
    if frame == 0:
        return np.array([s is State.EXPOSED for s in current], dtype=bool)
    previous = state_frames[frame - 1]
    return np.array([
        current[i] is State.EXPOSED and previous[i] is not State.EXPOSED
        for i in range(len(current))
    ], dtype=bool)


def _face_colors_with_fade(states: List[State],
                           recovered_ages: np.ndarray) -> np.ndarray:
    """Per-node RGBA face colours: normal state colours, recovered ones faded."""
    colors = np.zeros((len(states), 4))
    for i, s in enumerate(states):
        if s is State.RECOVERED:
            frac = min(1.0, float(recovered_ages[i]) / RECOVERED_FADE_DAYS)
            alpha = 1.0 - frac * (1.0 - RECOVERED_MIN_ALPHA)
            colors[i] = to_rgba(STATE_COLOR[s], alpha)
        else:
            colors[i] = to_rgba(STATE_COLOR[s])
    return colors


def _edge_styling(n: int, newly_infected: np.ndarray,
                  traveling: Optional[np.ndarray] = None):
    """Per-node marker edge colours/widths highlighting new cases and travelers."""
    edgecolors = np.full(n, "#555555", dtype=object)
    linewidths = np.full(n, 0.5)
    if traveling is not None:
        edgecolors[traveling] = TRAVELER_RING_COLOR
        linewidths[traveling] = 1.6
    edgecolors[newly_infected] = NEWLY_INFECTED_RING_COLOR
    linewidths[newly_infected] = 2.4
    return list(edgecolors), list(linewidths)


def _draw_transmission_flashes(ax, coords: np.ndarray,
                               transmission_frames: List[List],
                               frame: int, fig=None):
    """Return short-lived line artists for recent in-city transmissions.

    Mirrors the inter-city "string" fade already used for cross-city links,
    at the scale of one city's own node positions.
    """
    artists = []
    for age in range(min(frame + 1, TRANSMISSION_FADE_FRAMES + 1)):
        day = frame - age
        if day < 0 or day >= len(transmission_frames):
            continue
        alpha = 0.9 * (1.0 - age / (TRANSMISSION_FADE_FRAMES + 1))
        for source, target in transmission_frames[day]:
            line = ax.plot(
                [coords[source, 0], coords[target, 0]],
                [coords[source, 1], coords[target, 1]],
                color=NEWLY_INFECTED_RING_COLOR, linewidth=1.6,
                alpha=max(0.05, alpha), zorder=4, solid_capstyle="round",
            )[0]
            artists.append(line)
    return artists


def animate_states(state_frames: List[List[State]], history: List[DailyRecord],
                   config: Config, layout: str = "grid",
                   save_path: Optional[str] = None,
                   show: bool = False,
                   interval_ms: int = 400,
                   graph=None,
                   transmission_frames: Optional[List[List]] = None,
                   cluster_of: Optional[np.ndarray] = None,
                   ) -> FuncAnimation:
    """Animate per-individual state changes on a fixed layout.

    Beyond simple S/E/I/R colouring, a node gets a gold ring the day it's
    newly exposed, a brief flash line to whoever infected it, and recovered
    nodes gradually fade (lower alpha) the longer they've been immune.
    Positions never move -- only colour and these highlights change.

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
        graph: Optional persistent contact graph to render behind the nodes.
        transmission_frames: Per-day ``(source_id, target_id)`` exposure pairs
            (from :attr:`~simulation.Simulation.transmission_frames`), used
            for the transmission flash. ``None`` disables the effect.
        cluster_of: Per-individual cluster id from a ``clustered`` contact
            model, or ``None``. When given, overrides ``layout`` with
            :func:`cluster_layout` so clusters are visually obvious.

    Returns:
        The :class:`~matplotlib.animation.FuncAnimation`. Keep a reference to it
        alive until display/saving completes.
    """
    n = config.population_size
    positions = _resolve_layout(n, layout, cluster_of)
    coords = np.array(positions)
    recovered_ages = _recovered_run_lengths(state_frames)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("SEIR epidemic contact network of "
                 f"{n} people", fontsize=13, weight="bold")

    _draw_network_edges(ax, graph, coords)

    marker_size = max(90, min(600, 12000 / n))
    edgecolors0, linewidths0 = _edge_styling(
        n, _newly_infected_mask(state_frames, 0))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], s=marker_size,
        c=_face_colors_with_fade(state_frames[0], recovered_ages[0]),
        edgecolors=edgecolors0, linewidths=linewidths0, zorder=2)

    ax.set_aspect("equal")
    ax.axis("off")
    pad = 1.0
    ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)
    ax.legend(handles=_legend_handles(), loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False)

    day_text = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top",
                       ha="left", family="monospace", fontsize=10)
    flash_artists: List = []

    def update(frame: int):
        """Recolour every marker to its state on day ``frame``."""
        nonlocal flash_artists
        states = state_frames[frame]
        newly_infected = _newly_infected_mask(state_frames, frame)
        scatter.set_facecolor(_face_colors_with_fade(states, recovered_ages[frame]))
        edgecolors, linewidths = _edge_styling(n, newly_infected)
        scatter.set_edgecolors(edgecolors)
        scatter.set_linewidths(linewidths)

        while flash_artists:
            flash_artists.pop().remove()
        if transmission_frames:
            flash_artists = _draw_transmission_flashes(
                ax, coords, transmission_frames, frame)

        rec = history[frame]
        day_text.set_text(
            f"Day {rec.day}\n"
            f"S={rec.susceptible}  E={rec.exposed}  "
            f"I={rec.infectious}  R={rec.recovered}")
        return [scatter, day_text] + flash_artists

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
        layout: ``"grid"`` (default) or ``"circle"`` positioning; a city
            running the ``clustered`` contact model uses :func:`cluster_layout`
            instead, regardless of this setting, so its clusters stay visible.
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
    sizes = [c.config.population_size for c in cities]

    # Per-day travel counts between each ordered city pair, for the arrows.
    travel_by_day: Dict[int, Dict[Tuple[int, int], int]] = defaultdict(
        lambda: defaultdict(int))
    for te in regional_sim.travel_events:
        travel_by_day[te.day][(te.home_city_id, te.destination_city_id)] += 1

    cluster_of_per_city = [
        getattr(city.engine.contact_model, "cluster_of", None) for city in cities
    ]
    positions_per_city = [
        _resolve_layout(n, layout, cluster_of)
        for n, cluster_of in zip(sizes, cluster_of_per_city)
    ]

    fig, axes = plt.subplots(1, num_cities, figsize=(6*num_cities, 5))
    if num_cities == 1:
        axes = [axes]

    fig.suptitle(
        f"Regional SEIR contact networks ({num_cities} cities)",
        fontsize=13, weight="bold"
    )

    scatters = []
    day_texts = []
    coords_per_city: List[np.ndarray] = []
    recovered_ages_per_city = [_recovered_run_lengths(city.state_frames)
                              for city in cities]

    for city_idx, (ax, city, positions) in enumerate(
        zip(axes, cities, positions_per_city)
    ):
        coords = np.array(positions)
        coords_per_city.append(coords)
        _draw_network_edges(ax, city.network, coords)
        marker_size = max(40, min(600, 12000 / max(1, sizes[city_idx])))
        edgecolors0, linewidths0 = _edge_styling(
            sizes[city_idx], _newly_infected_mask(city.state_frames, 0))
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1], s=marker_size,
            c=_face_colors_with_fade(city.state_frames[0],
                                     recovered_ages_per_city[city_idx][0]),
            edgecolors=edgecolors0, linewidths=linewidths0, zorder=2
        )
        scatters.append(scatter)

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"City {city_label(city_idx)} (n={sizes[city_idx]})",
                     fontsize=11, weight="bold")
        pad = 1.15
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
    cross_city_artists: List = []

    # Figure-fraction center of each city panel, used as arrow endpoints so
    # travel between ANY pair of cities is visible -- not just neighbours in
    # the row layout. Asymmetric travel matrices routinely connect non-adjacent
    # cities (e.g. city 0 -> city 3 while skipping 1 and 2)
    panel_centers = []
    for ax in axes:
        pos = ax.get_position()
        panel_centers.append(((pos.x0 + pos.x1) / 2, (pos.y0 + pos.y1) / 2))

    def _draw_travel(frame: int) -> None:
        """Draw a dashed arrow between every city pair with travel today."""
        while travel_artists:
            travel_artists.pop().remove()
        if not show_travel or num_cities < 2:
            return
        day_travel = travel_by_day.get(frame, {})
        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                count = day_travel.get((i, j), 0) + day_travel.get((j, i), 0)
                if count == 0:
                    continue
                x0, y0 = panel_centers[i]
                x1, y1 = panel_centers[j]
                # Adjacent panels get a straight connector; non-adjacent pairs
                # arc up and over the panels in between so the line doesn't
                # cut through unrelated cities.
                rad = 0.0 if j == i + 1 else 0.25 + 0.05 * (j - i)
                arrow = FancyArrowPatch(
                    (x0, y0), (x1, y1), transform=fig.transFigure,
                    arrowstyle="<|-|>", mutation_scale=13, linewidth=1.3,
                    linestyle=(0, (4, 3)), color="#4062bb", zorder=5,
                    connectionstyle=f"arc3,rad={rad}",
                    shrinkA=40, shrinkB=40)
                fig.add_artist(arrow)
                label_y = max(y0, y1) + 0.06 + 0.05 * (j - i - 1)
                label = fig.text((x0 + x1) / 2, label_y, f"{count} trav.",
                                 ha="center", va="bottom", fontsize=8,
                                 color="#4062bb", weight="bold")
                travel_artists.extend([arrow, label])

    # How many days a transmission "string" stays visible before fully fading,
    # so the animation highlights *recent* transmission events (the fact that
    # explains why a city just started an outbreak) rather than accumulating
    # an ever-growing, increasingly unreadable tangle over a long run.
    FADE_WINDOW_DAYS = 10

    def _draw_cross_city_links(frame: int) -> None:
        """Draw fading strings for recent travel-caused transmissions."""
        while cross_city_artists:
            cross_city_artists.pop().remove()
        for link in regional_sim.intercity_transmissions:
            age = frame - link.day
            if age < 0 or age > FADE_WINDOW_DAYS:
                continue
            alpha = 0.85 * (1.0 - age / FADE_WINDOW_DAYS)
            source_axes = axes[link.source_city_id]
            target_axes = axes[link.target_city_id]
            source_xy = coords_per_city[link.source_city_id][link.source_individual_id]
            target_xy = coords_per_city[link.target_city_id][link.target_individual_id]
            source_fig = fig.transFigure.inverted().transform(
                source_axes.transData.transform(source_xy))
            target_fig = fig.transFigure.inverted().transform(target_axes.transData.transform(target_xy))
            string = FancyArrowPatch(
                source_fig, target_fig, transform=fig.transFigure,
                arrowstyle="->", mutation_scale=10, linewidth=1.4,
                color="#6c63ff", alpha=max(0.05, alpha), zorder=6,
            )
            fig.add_artist(string)
            cross_city_artists.append(string)

    transmission_artists: List[List] = [[] for _ in cities]

    def update(frame: int):
        """Recolour every marker to its state on day ``frame``."""
        for city_idx, city in enumerate(cities):
            if frame < len(city.state_frames):
                states = city.state_frames[frame]
                newly_infected = _newly_infected_mask(city.state_frames, frame)
                traveling = None
                if frame < len(city.travel_status_frames):
                    away_ids = city.travel_status_frames[frame].keys()
                    traveling = np.zeros(len(states), dtype=bool)
                    for away_id in away_ids:
                        traveling[away_id] = True
                scatters[city_idx].set_facecolor(_face_colors_with_fade(
                    states, recovered_ages_per_city[city_idx][frame]))
                edgecolors, linewidths = _edge_styling(
                    len(states), newly_infected, traveling)
                scatters[city_idx].set_edgecolors(edgecolors)
                scatters[city_idx].set_linewidths(linewidths)

                while transmission_artists[city_idx]:
                    transmission_artists[city_idx].pop().remove()
                if frame < len(city.transmission_frames):
                    transmission_artists[city_idx] = _draw_transmission_flashes(
                        axes[city_idx], coords_per_city[city_idx],
                        city.transmission_frames, frame)

                if frame < len(city.history):
                    rec = city.history[frame]
                    isolated_txt = "  [ISOLATED]" if city.isolated else ""
                    day_texts[city_idx].set_text(
                        f"Day {rec.day}{isolated_txt}\n"
                        f"S={rec.susceptible} E={rec.exposed}\n"
                        f"I={rec.infectious} R={rec.recovered}"
                    )
        _draw_travel(frame)
        _draw_cross_city_links(frame)
        return (scatters + day_texts + travel_artists + cross_city_artists
               + [a for artists in transmission_artists for a in artists])

    max_frames = max(len(city.state_frames) for city in cities)
    anim = FuncAnimation(
        fig, update, frames=max_frames,
        interval=interval_ms, blit=False, repeat=False
    )

    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.subplots_adjust(wspace=0.35)
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

        ax.set_title(f"City {city_label(city_idx)} (n={city.config.population_size})",
                     fontsize=11, weight="bold")
        ax.set_xlabel("Day")
        ax.set_ylabel("Individuals")
        ax.set_ylim(0, city.config.population_size)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"Saved regional curves to {save_path}")
    if show:
        plt.show()
    return fig


# 
# Mode 2: population-tile heat map -- one city block per city
# 
def _population_tile_groups(population: int, tile_fraction: float) -> List[np.ndarray]:
    """Partition a city's residents into stable, equally sized visual cohorts.

    The number of tiles scales with population size while preserving a
    configurable approximate size per tile. A 5% tile fraction gives roughly 20
    tiles for a city of 400 people, and fewer for smaller cities.
    """
    desired_tiles = max(1, int(round(1 / max(tile_fraction, 1e-6))))
    tile_count = min(population, max(1, desired_tiles))
    if population <= 4:
        return [np.array([i], dtype=int) for i in range(population)]
    return list(np.array_split(np.arange(population), tile_count))


def animate_regional_heatmap(regional_sim: "RegionalSimulation",
                             save_path: Optional[str] = None,
                             show: bool = False,
                             interval_ms: int = 400,
                             heatmap_tile_fraction: float = 0.05,
                             **_ignored) -> FuncAnimation:
    """Animate cities as blocks of population tiles coloured by % infectious.

    Every square represents approximately ``heatmap_tile_fraction`` of one
    city's population. Its membership is fixed for the complete animation and
    its colour uses the same infectious-share measure as the prior circle view.
    """
    cities = regional_sim.cities
    n = len(cities)
    num_frames = max(len(city.history) for city in cities)
    city_groups = [_population_tile_groups(city.config.population_size,
                                            heatmap_tile_fraction)
                   for city in cities]
    tile_fracs: List[np.ndarray] = []
    for city, groups in zip(cities, city_groups):
        values = np.zeros((num_frames, len(groups)))
        for frame in range(num_frames):
            states = city.state_frames[min(frame, len(city.state_frames) - 1)]
            infectious = np.fromiter((state == State.INFECTIOUS for state in states),
                                     dtype=bool, count=len(states))
            values[frame] = [infectious[group].mean() for group in groups]
        tile_fracs.append(values)
    vmax = max(0.05, max(float(values.max()) for values in tile_fracs))
    try:
        cmap = plt.get_cmap("YlOrRd")
    except AttributeError:
        cmap = plt.cm.get_cmap("YlOrRd")

    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, squeeze=False,
                             figsize=(3.25 * cols, 3.45 * rows))
    flat_axes = axes.ravel()
    fig.suptitle("Regional population-tile heat map", fontsize=14, weight="bold")
    scatters = []
    for city_idx, (ax, city, groups, values) in enumerate(
            zip(flat_axes, cities, city_groups, tile_fracs)):
        tile_count = len(groups)
        tile_cols = int(np.ceil(np.sqrt(tile_count)))
        tile_rows = int(np.ceil(tile_count / tile_cols))
        coords = np.array([(i % tile_cols, -(i // tile_cols))
                           for i in range(tile_count)], dtype=float)
        marker_size = max(120, min(860, 7000 / max(tile_cols, tile_rows) ** 2))
        scatter = ax.scatter(coords[:, 0], coords[:, 1], s=marker_size,
                             marker="s", c=values[0], cmap=cmap,
                             vmin=0.0, vmax=vmax, edgecolors="white",
                             linewidths=1.2)
        scatters.append(scatter)
        ax.set_title(f"City {city_label(city_idx)} (n={city.config.population_size})",
                     fontsize=11, weight="bold")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-0.7, tile_cols - 0.3)
        ax.set_ylim(-tile_rows + 0.3, 0.7)
        representative_size = city.config.population_size / tile_count
        ax.text(0.5, -0.10, f"{tile_count} tiles · ≈{representative_size:.0f} people/tile",
                transform=ax.transAxes, ha="center", va="top", fontsize=8,
                color="#555555")
    for ax in flat_axes[n:]:
        ax.axis("off")
    cbar = fig.colorbar(scatters[0], ax=flat_axes[:n].tolist(), location="right",
                        fraction=0.025, pad=0.03)
    cbar.set_label("infectious share within tile", fontsize=9)
    cbar.ax.set_title("0%  25%  50%  75%  100%", fontsize=8)
    fig.text(0.01, 0.01,
             "Tiles are fixed resident cohorts, not geographic neighbourhoods.",
             fontsize=8, color="#555555")
    day_text = fig.text(0.01, 0.965, "", va="top", ha="left",
                        family="monospace", fontsize=10)

    def update(frame: int):
        for scatter, values in zip(scatters, tile_fracs):
            scatter.set_array(values[frame])
        day_text.set_text(f"Day {frame}")
        return scatters + [day_text]

    anim = FuncAnimation(fig, update, frames=num_frames,
                         interval=interval_ms, blit=False, repeat=False)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.90, bottom=0.08)
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=max(1, round(1000 / interval_ms))))
        print(f"Saved regional heat map to {save_path}")
    if show:
        plt.show()
    return anim


# 
# Mode 3: per-city pie charts of the S/E/I/R breakdown, animated
# 
def animate_regional_pies(regional_sim: "RegionalSimulation",
                          save_path: Optional[str] = None,
                          show: bool = False,
                          interval_ms: int = 400,
                          **_ignored) -> FuncAnimation:
    """Animate each city as a pie chart of its S/E/I/R composition over time."""
    cities = regional_sim.cities
    n = len(cities)
    num_frames = len(cities[0].history)

    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    axes = np.array(axes).reshape(-1)
    fig.suptitle("Regional SEIR composition by city", fontsize=13, weight="bold")

    order = [State.SUSCEPTIBLE, State.EXPOSED, State.INFECTIOUS, State.RECOVERED]
    colors = [STATE_COLOR[s] for s in order]

    def update(frame: int):
        for j, city in enumerate(cities):
            ax = axes[j]
            ax.clear()
            rec = city.history[min(frame, len(city.history) - 1)]
            values = [rec.susceptible, rec.exposed, rec.infectious, rec.recovered]
            if sum(values) == 0:
                values = [1, 0, 0, 0]
            ax.pie(values, colors=colors, startangle=90,
                   wedgeprops={"edgecolor": "white", "linewidth": 0.5})
            ax.set_aspect("equal")
            ax.set_title(f"City {city_label(j)} (n={city.config.population_size})",
                         fontsize=10, weight="bold")
        for k in range(n, len(axes)):
            axes[k].axis("off")
        fig.text(0.01, 0.99, f"Day {frame}", va="top", ha="left",
                 family="monospace", fontsize=10)
        return list(axes)

    anim = FuncAnimation(fig, update, frames=num_frames,
                         interval=interval_ms, blit=False, repeat=False)
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=4,
               frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=max(1, round(1000 / interval_ms))))
        print(f"Saved regional pie charts to {save_path}")
    if show:
        plt.show()
    return anim


# 
# Mode "cluster": communities within each city, for medium populations
# 
def _detect_communities(graph, population_size: int) -> List[List[int]]:
    """Partition a city's population into communities for the cluster view.

    Uses greedy modularity community detection on the persistent contact graph
    (so clusters reflect real social structure -- neighbourhoods, workplaces).
    Falls back to a handful of equal-sized synthetic groups when there is no
    graph to partition (the well-mixed model has none), so the mode still
    degrades gracefully rather than failing.

    Args:
        graph: The city's contact graph, or ``None``.
        population_size: Number of individuals, used for the fallback split.

    Returns:
        A list of communities, each a list of individual ids.
    """
    if graph is not None and graph.number_of_edges() > 0:
        communities = nx.algorithms.community.greedy_modularity_communities(graph)
        return [sorted(c) for c in communities]
    num_groups = max(1, int(np.ceil(np.sqrt(population_size))))
    ids = np.arange(population_size)
    return [list(chunk) for chunk in np.array_split(ids, num_groups) if len(chunk)]


def animate_regional_clusters(regional_sim: "RegionalSimulation",
                              save_path: Optional[str] = None,
                              show: bool = False,
                              interval_ms: int = 400,
                              **_ignored) -> FuncAnimation:
    """Animate each city as a graph of its own social communities.

    Individual nodes are collapsed into communities (detected once from the
    persistent contact network), each drawn as one bubble sized by its
    population and coloured by its current fraction infectious+exposed, with
    edges showing how strongly two communities are connected. This keeps local
    spread visible -- you can watch an outbreak move from one neighbourhood's
    bubble to the next -- without drawing every one of a few hundred nodes.

    Cities remain separate, spatially distinct subplots, exactly as in the
    network view.
    """
    cities = regional_sim.cities
    num_cities = len(cities)
    num_frames = len(cities[0].history)

    fig, axes = plt.subplots(1, num_cities, figsize=(6 * num_cities, 5))
    if num_cities == 1:
        axes = [axes]
    fig.suptitle(f"Regional SEIR by community ({num_cities} cities)",
                 fontsize=13, weight="bold")

    scatters = []
    day_texts = []
    community_members: List[List[List[int]]] = []  # per city, per community

    for city_idx, (ax, city) in enumerate(zip(axes, cities)):
        communities = _detect_communities(city.network, city.config.population_size)
        community_members.append(communities)
        sizes = np.array([len(c) for c in communities], dtype=float)

        # Lay communities out with a spring layout on a coarsened graph whose
        # edge weight is the number of contact-graph edges between the two
        # communities, so tightly-linked communities are drawn close together.
        coarse = nx.Graph()
        coarse.add_nodes_from(range(len(communities)))
        if city.network is not None:
            owner = {}
            for ci, members in enumerate(communities):
                for m in members:
                    owner[m] = ci
            for u, v in city.network.edges():
                cu, cv = owner[u], owner[v]
                if cu != cv:
                    w = coarse.get_edge_data(cu, cv, {}).get("weight", 0) + 1
                    coarse.add_edge(cu, cv, weight=w)
        pos = nx.spring_layout(coarse, seed=42, weight="weight") if len(communities) > 1 \
            else {0: (0.0, 0.0)}
        coords = np.array([pos[i] for i in range(len(communities))])

        if coarse.number_of_edges() > 0:
            segments = [(coords[u], coords[v]) for u, v in coarse.edges()]
            ax.add_collection(LineCollection(
                segments, colors="#9aa0a6", linewidths=1.0, alpha=0.6, zorder=1))

        marker = 200 + 3000 * (sizes / sizes.max())
        scatter = ax.scatter(coords[:, 0], coords[:, 1], s=marker,
                             c=[0.0] * len(communities), cmap="OrRd",
                             vmin=0.0, vmax=1.0,
                             edgecolors="#333333", linewidths=1.0, zorder=2)
        scatters.append(scatter)

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"City {city_label(city_idx)} (n={city.config.population_size}, "
                     f"{len(communities)} communities)", fontsize=10, weight="bold")
        pad = 0.4
        ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
        ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)

        day_text = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top",
                           ha="left", family="monospace", fontsize=9)
        day_texts.append(day_text)

    fig.colorbar(scatters[0], ax=list(axes), fraction=0.03, pad=0.02,
                 label="fraction infectious + exposed", shrink=0.8)
    fig.subplots_adjust(right=0.92)

    def update(frame: int):
        for city_idx, city in enumerate(cities):
            f = min(frame, len(city.state_frames) - 1)
            states = city.state_frames[f]
            fracs = []
            for members in community_members[city_idx]:
                sick = sum(1 for m in members
                          if states[m] in (State.EXPOSED, State.INFECTIOUS))
                fracs.append(sick / len(members) if members else 0.0)
            scatters[city_idx].set_array(np.array(fracs))
            if frame < len(city.history):
                rec = city.history[frame]
                day_texts[city_idx].set_text(
                    f"Day {rec.day}\nS={rec.susceptible} E={rec.exposed}\n"
                    f"I={rec.infectious} R={rec.recovered}")
        return scatters + day_texts

    anim = FuncAnimation(fig, update, frames=num_frames,
                         interval=interval_ms, blit=False, repeat=False)
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=max(1, round(1000 / interval_ms))))
        print(f"Saved regional cluster view to {save_path}")
    if show:
        plt.show()
    return anim


# 
# Dispatcher
# 
def resolve_visualization_mode(regional_sim: "RegionalSimulation", mode: str) -> str:
    """Resolve ``"auto"`` to a concrete mode based on the largest city.

    Args:
        regional_sim: The simulation whose cities determine the auto choice.
        mode: The requested mode; passed through unchanged unless ``"auto"``.

    Returns:
        A concrete mode: ``"network"``, ``"cluster"``, or ``"heatmap"``.
    """
    if mode != "auto":
        return mode
    max_pop = max(c.config.population_size for c in regional_sim.cities)
    if max_pop <= NETWORK_MAX_POPULATION:
        return "network"
    if max_pop <= CLUSTER_MAX_POPULATION:
        return "cluster"
    return "heatmap"


def animate_regional(regional_sim: "RegionalSimulation", mode: str = "auto",
                     layout: str = "grid", save_path: Optional[str] = None,
                     show: bool = False, interval_ms: int = 400,
                     heatmap_tile_fraction: float = 0.05) -> FuncAnimation:
    """Animate a regional run in the requested (or auto-detected) mode.

    Args:
        regional_sim: The completed simulation.
        mode: ``"auto"`` (pick from population size), ``"network"``
            (per-individual nodes; <=150/city), ``"cluster"`` (communities;
            <=1000/city), ``"heatmap"`` (population tiles; any size), or
            ``"pie"`` (S/E/I/R composition per city).
        layout: Node layout for the network mode.
        save_path/show/interval_ms: Standard animation output controls.

    Returns:
        The created :class:`~matplotlib.animation.FuncAnimation`.
    """
    resolved = resolve_visualization_mode(regional_sim, mode)
    if resolved != mode:
        print(f"[viz] auto-selected '{resolved}' mode "
              f"(largest city: {max(c.config.population_size for c in regional_sim.cities)} people)")

    if resolved == "heatmap":
        return animate_regional_heatmap(
            regional_sim, save_path=save_path, show=show, interval_ms=interval_ms,
            heatmap_tile_fraction=heatmap_tile_fraction)
    if resolved == "cluster":
        return animate_regional_clusters(
            regional_sim, save_path=save_path, show=show, interval_ms=interval_ms)
    if resolved == "pie":
        return animate_regional_pies(
            regional_sim, save_path=save_path, show=show, interval_ms=interval_ms)
    if resolved != "network":
        raise ValueError(f"Unknown visualization mode: {mode}")
    return animate_regional_states(
        regional_sim, layout=layout, save_path=save_path, show=show,
        interval_ms=interval_ms)
