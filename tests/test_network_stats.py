import networkx as nx
import numpy as np

from config import Config
from interaction import RandomNetworkContactModel, network_topology_report
from regional_simulation import RegionalSimulation
from simulation import Simulation


def test_network_topology_report_on_connected_ring():
    graph = nx.cycle_graph(10)
    report = network_topology_report(graph)
    assert report["node_count"] == 10
    assert report["edge_count"] == 10
    assert report["mean_degree"] == 2.0
    assert report["num_connected_components"] == 1
    assert report["largest_component_size"] == 10
    assert report["largest_component_fraction"] == 1.0
    assert report["path_length_computed"] is True
    # A 10-node ring's average shortest path length is a known constant.
    assert abs(report["average_shortest_path_length"] - nx.average_shortest_path_length(graph)) < 1e-9


def test_network_topology_report_on_disconnected_graph():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)])  # triangle, 3 nodes
    graph.add_nodes_from([3, 4])  # two isolated nodes
    report = network_topology_report(graph)
    assert report["node_count"] == 5
    assert report["num_connected_components"] == 3
    assert report["largest_component_size"] == 3
    assert report["largest_component_fraction"] == 3 / 5
    # Path length should be computed on the 3-node triangle component only.
    assert report["path_length_computed"] is True
    assert "largest connected component" in report["path_length_note"]


def test_network_topology_report_skips_path_length_above_node_cap():
    graph = nx.cycle_graph(20)
    report = network_topology_report(graph, path_length_node_cap=10)
    assert report["path_length_computed"] is False
    assert np.isnan(report["average_shortest_path_length"])
    assert "skipped" in report["path_length_note"]


def test_network_topology_report_on_all_isolated_nodes():
    graph = nx.Graph()
    graph.add_nodes_from(range(5))
    report = network_topology_report(graph)
    assert report["num_connected_components"] == 5
    assert report["largest_component_size"] == 1
    assert report["path_length_computed"] is False


def test_city_and_simulation_network_report_match_underlying_graph():
    config = Config(population_size=30, contact_model="random-network",
                    random_degree_min=2, random_degree_max=5, random_seed=7)
    sim = Simulation(config)
    report = sim.network_report()
    model = sim.engine.contact_model
    assert isinstance(model, RandomNetworkContactModel)
    assert report["node_count"] == 30
    assert report["edge_count"] == model.graph.number_of_edges()


def test_well_mixed_network_report_is_empty_no_graph():
    config = Config(population_size=20, contact_model="well-mixed",
                    daily_contacts=4, random_seed=1)
    sim = Simulation(config)
    assert sim.network_report() == {}


def test_regional_city_network_report_available_per_city():
    config = Config(city_populations=(20, 20), clustered_cities=(0,),
                    contact_model="daily-random", num_clusters=4,
                    random_seed=3, simulation_days=5)
    regional = RegionalSimulation(config)
    report_a = regional.cities[0].network_report()
    report_b = regional.cities[1].network_report()
    assert report_a["node_count"] == 20
    assert report_b["node_count"] == 20
