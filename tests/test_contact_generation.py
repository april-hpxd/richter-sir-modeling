import numpy as np

from config import Config
from interaction import ClusteredContactModel, DailyRandomContactModel
from simulation import Simulation


def test_clustered_contacts_have_no_self_or_duplicate_ids():
    model = ClusteredContactModel(
        population_size=50, num_clusters=10,
        within_cluster_contact_probability=0.9,
        min_degree=2, max_degree=7, rng=np.random.default_rng(12),
    )
    for _ in range(5):
        model.prepare_day(np.random.default_rng(20 + _))
        for person_id, contacts in enumerate(model.contact_lists):
            assert person_id not in contacts
            assert len(contacts) == len(set(contacts))


def test_clustered_locality_and_daily_random_rate_are_separate_controls():
    clustered = ClusteredContactModel(
        population_size=100, num_clusters=10,
        within_cluster_contact_probability=0.9,
        min_degree=2, max_degree=7, rng=np.random.default_rng(1),
    )
    regular = DailyRandomContactModel(
        population_size=100, min_degree=2, max_degree=7,
        rng=np.random.default_rng(1),
    )
    for day in range(30):
        clustered.prepare_day(np.random.default_rng(100 + day))
        regular.prepare_day(np.random.default_rng(200 + day))
    clustered_stats = clustered.mean_contact_stats()
    regular_stats = regular.mean_contact_stats()
    assert abs(clustered_stats["average_contacts"] -
               regular_stats["average_contacts"]) < 0.3
    assert clustered_stats["within_cluster_contact_proportion"] > 0.8


def test_same_seed_reproduces_clustered_simulation():
    config = Config(population_size=50, contact_model="clustered",
                    num_clusters=5, within_cluster_contact_probability=0.9,
                    simulation_days=20, random_seed=44)
    first = Simulation(config).run()
    second = Simulation(config).run()
    assert first == second


def test_clustered_configuration_handles_singleton_clusters():
    model = ClusteredContactModel(
        population_size=5, num_clusters=5,
        within_cluster_contact_probability=1.0,
        min_degree=1, max_degree=2, rng=np.random.default_rng(3),
    )
    for contacts in model.contact_lists:
        assert len(contacts) >= 1
        assert len(contacts) == len(set(contacts))