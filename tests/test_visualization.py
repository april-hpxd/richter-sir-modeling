import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from disease_model import State
from visualization import animate_regional_heatmap


class DummyConfig:
    def __init__(self, population_size: int):
        self.population_size = population_size


class DummyCity:
    def __init__(self, population_size: int, num_frames: int):
        self.config = DummyConfig(population_size)
        self.history = [object() for _ in range(num_frames)]
        self.state_frames = [
            [State.SUSCEPTIBLE] * population_size for _ in range(num_frames)
        ]


class DummyRegionalSimulation:
    def __init__(self, cities):
        self.cities = cities


def test_animate_regional_heatmap_runs_without_cmap_errors():
    sim = DummyRegionalSimulation([
        DummyCity(4, 2),
        DummyCity(6, 2),
    ])

    anim = animate_regional_heatmap(sim, interval_ms=50)

    assert anim is not None
    plt.close("all")
