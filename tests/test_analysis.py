from config import Config
from regional_simulation import RegionalSimulation
from analysis import analyze_network_importance, evaluate_interventions, InterventionSpec


def test_evaluate_interventions_returns_ranked_results():
    config = Config(
        population_size=20,
        number_of_cities=2,
        population_per_city=20,
        simulation_days=20,
        infection_probability=0.2,
        incubation_days=2,
        infectious_days=3,
        initial_infected=1,
        random_seed=7,
        travel_fraction=0.2,
        daily_travel_rate=0.3,
        contact_model="random-network",
        random_degree_min=2,
        random_degree_max=4,
    )

    interventions = [
        InterventionSpec(name="isolate_city_1", kind="isolate_city", city_ids=[1]),
        InterventionSpec(name="reduce_travel_globally", kind="reduce_travel_globally", factor=0.5),
    ]

    report = evaluate_interventions(
        config,
        interventions,
        num_runs=1,
        base_seed=0,
        verbose=False,
    )

    assert report["ranked_interventions"]
    assert len(report["ranked_interventions"]) == len(interventions)
    assert all(result["reduction_infections"] >= 0 for result in report["ranked_interventions"])


def test_analyze_network_importance_returns_city_rankings():
    config = Config(
        population_size=24,
        number_of_cities=2,
        population_per_city=24,
        simulation_days=18,
        infection_probability=0.18,
        incubation_days=2,
        infectious_days=3,
        initial_infected=1,
        random_seed=11,
        travel_fraction=0.15,
        daily_travel_rate=0.25,
        contact_model="random-network",
        random_degree_min=2,
        random_degree_max=4,
    )

    sim = RegionalSimulation(config)
    sim.run(verbose=False)

    report = analyze_network_importance(sim)

    assert report["city_metrics"]
    assert report["city_rankings"]
    assert len(report["city_rankings"]) == len(report["city_metrics"])
