import numpy as np

from city import City, CityConfig, DiseaseToken
from disease_model import State


def test_travel_acquired_exposure_starts_incubation_next_day():
    city_config = CityConfig(
        population_size=2,
        daily_contacts=1,
        infection_probability=1.0,
        incubation_days=2,
        infectious_days=5,
        contact_model_type="well-mixed",
        watts_strogatz_k=2,
        watts_strogatz_p=0.1,
        random_degree_min=1,
        random_degree_max=1,
    )
    city = City(0, city_config, np.random.default_rng(31))
    city.engine.individuals[0].state = State.INFECTIOUS
    token = DiseaseToken(state=State.SUSCEPTIBLE, days_in_state=0)

    city.host_visitor_day(token, np.random.default_rng(1), 1, "1-0")

    assert token.state is State.EXPOSED
    assert token.days_in_state == 0
