"""Regional simulation: multiple cities connected by temporary daily travel.

A :class:`RegionalSimulation` coordinates several independent :class:`~city.City`
objects and layers a simple *commuting* travel model on top. Each simulated day:

1. Disease progresses **inside** every city, through that city's own contact
   network (see :meth:`City.advance_disease`).
2. Today's travelers are chosen from each city's fixed pool of eligible
   commuters.
3. Each traveler visits a random other city and mingles with residents there,
   interacting through the **destination city's contact network**.
4. Transmission during the visit is bidirectional and carries the traveler's
   disease state with them:

   * an *infectious* traveler can expose susceptible residents of the
     destination (seeding the outbreak there);
   * a *susceptible* traveler can be exposed by infectious residents and brings
     that infection home.

5. Everyone returns home; imported infections are folded into the destination /
   home city's statistics for that same day.

This is deliberately the *simplest* thing that reproduces the phenomenon of
interest -- how travel delays or accelerates the arrival of an outbreak in
another city -- with no routes, schedules, or transport modes. All travel
randomness is drawn from the single master generator, so the whole regional run
is reproducible from one seed. Replacing this layer with a richer transport
model requires no change to the disease engine or to :class:`City`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from city import City, CityConfig
from config import Config
from disease_model import State


@dataclass
class TravelEvent:
    """Record of one traveler's day trip.

    Attributes:
        day: Simulated day of travel.
        traveler_id: The traveler's id in their home city.
        home_city_id: City the traveler started (and returns) to.
        destination_city_id: City they visited for the day.
        state_before: Traveler's disease state on departure.
        state_after: Traveler's disease state after returning home.
        acquired_infection: True if the traveler was exposed while away and
            carried the infection home.
        residents_infected: Number of destination residents this (infectious)
            traveler exposed during the visit.
    """

    day: int
    traveler_id: int
    home_city_id: int
    destination_city_id: int
    state_before: State
    state_after: State
    acquired_infection: bool
    residents_infected: int


class RegionalSimulation:
    """Coordinate multiple cities connected by temporary daily travel.

    Attributes:
        config: The global configuration.
        cities: List of :class:`~city.City` objects, indexed by city id.
        rng: The master generator seeding every city and driving all travel.
        eligible: Per-city fixed list of ids allowed to travel (the commuters).
        travel_events: Every :class:`TravelEvent` recorded across the run.
        history: One regional-statistics dict per day (including day 0).
    """

    def __init__(self, config: Config) -> None:
        """Build the cities, choose commuters, seed City A, and record day 0.

        Creates ``config.number_of_cities`` independent cities, each with its
        own RNG spawned from the master seed. City 0 ("City A") is seeded with
        ``config.initial_infected`` exposed individuals; every other city starts
        fully susceptible, so the arrival day of infection elsewhere is
        meaningful.

        Args:
            config: Global configuration (disease, network, and travel params).
        """
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

        self.cities: List[City] = []
        for city_id in range(config.number_of_cities):
            city_rng = np.random.default_rng(int(self.rng.integers(0, 2**31)))
            city_config = CityConfig(
                population_size=config.population_per_city,
                daily_contacts=config.daily_contacts,
                infection_probability=config.infection_probability,
                incubation_days=config.incubation_days,
                infectious_days=config.infectious_days,
                contact_model_type=config.contact_model,
                watts_strogatz_k=config.watts_strogatz_k,
                watts_strogatz_p=config.watts_strogatz_p,
            )
            self.cities.append(City(city_id, city_config, city_rng))

        # Fixed pool of eligible commuters per city (a property of residents,
        # chosen once from the master stream), from which a fraction travel
        # each day. Choosing it up front keeps "who commutes" stable over time.
        self.eligible: List[np.ndarray] = []
        pool_size = int(config.population_per_city * config.travel_fraction)
        for _ in self.cities:
            if pool_size > 0:
                pool = self.rng.choice(
                    config.population_per_city, size=pool_size, replace=False)
            else:
                pool = np.empty(0, dtype=np.int64)
            self.eligible.append(np.sort(pool))

        self.travel_events: List[TravelEvent] = []
        self.history: List[Dict] = []
        self._day = 0

        # Seed the outbreak in City A only, then record the day-0 baseline for
        # every city so histories/animation frames are aligned from day 0.
        seeded = config.initial_infected
        self.cities[0].seed_infection(seeded)
        for city in self.cities:
            city.record_day(
                new_exposed=(seeded if city.id == 0 else 0),
                new_infectious=0,
                new_recovered=0,
            )
        self.history.append(self._compute_regional_stats(
            num_travelers=0, new_imported=0))

    # ------------------------------------------------------------------
    # Daily update
    # ------------------------------------------------------------------
    def step(self) -> Dict:
        """Advance the whole region by one day and record regional statistics.

        Order of operations (matching the milestone specification):

        1. Advance disease inside every city (internal network only).
        2. Choose today's travelers and their destinations.
        3. Travelers interact via the destination city's contact network;
           infections spread in both directions and stay with each person.
        4. Record each city's day, folding travel-imported exposures into that
           day's ``new_exposed`` so they appear in the same day's stats/frames.

        Returns:
            The regional-statistics dict for the day just simulated.
        """
        self._day += 1

        # Step 1: internal disease dynamics for every city (not yet recorded).
        deltas = [city.advance_disease() for city in self.cities]

        # Steps 2-3: travel and bidirectional, network-based transmission.
        # ``imported_today[c]`` counts exposures created in city ``c`` by travel.
        imported_today = [0] * len(self.cities)
        travelers = self._select_travelers()
        for home_id, traveler_id, dest_id in travelers:
            self._execute_trip(home_id, int(traveler_id), dest_id,
                               imported_today)

        # Step 4: record each city with internal + imported new exposures.
        for city, delta, imported in zip(self.cities, deltas, imported_today):
            city.record_day(
                new_exposed=delta["new_exposed"] + imported,
                new_infectious=delta["new_infectious"],
                new_recovered=delta["new_recovered"],
            )

        stats = self._compute_regional_stats(
            num_travelers=len(travelers), new_imported=sum(imported_today))
        self.history.append(stats)
        return stats

    def run(self, verbose: bool = False) -> List[Dict]:
        """Run the region to completion.

        Stops once no city has any ``EXPOSED`` or ``INFECTIOUS`` individual
        (nothing can change), or after ``config.simulation_days`` days.

        Args:
            verbose: If True, print a one-line regional summary each day.

        Returns:
            The full list of daily regional-statistics dicts.
        """
        for _ in range(self.config.simulation_days):
            stats = self.step()
            if verbose:
                print(
                    f"Day {stats['day']:3d} | "
                    f"S={stats['total_susceptible']:3d} "
                    f"E={stats['total_exposed']:3d} "
                    f"I={stats['total_infectious']:3d} "
                    f"R={stats['total_recovered']:3d} | "
                    f"travelers={stats['num_travelers']:2d} "
                    f"imported={stats['new_imported']}"
                )
            if all(not city.is_epidemic_active() for city in self.cities):
                break
        return self.history

    # ------------------------------------------------------------------
    # Travel internals
    # ------------------------------------------------------------------
    def _select_travelers(self) -> List[Tuple[int, int, int]]:
        """Choose today's travelers and assign each a destination city.

        From each city's fixed eligible pool, ``daily_travel_rate`` of the pool
        is selected (rounded) to travel today. Every traveler goes to a random
        *other* city. All draws come from the master generator.

        Returns:
            A list of ``(home_city_id, traveler_id, destination_city_id)``.
        """
        travelers: List[Tuple[int, int, int]] = []
        num_cities = len(self.cities)
        if num_cities < 2:
            return travelers  # nowhere to travel

        for home_id, pool in enumerate(self.eligible):
            if len(pool) == 0:
                continue
            n_travel = int(round(len(pool) * self.config.daily_travel_rate))
            n_travel = min(n_travel, len(pool))
            if n_travel <= 0:
                continue
            chosen = self.rng.choice(pool, size=n_travel, replace=False)
            for traveler_id in chosen:
                # Uniformly pick one of the other cities as the destination.
                offset = int(self.rng.integers(1, num_cities))
                dest_id = (home_id + offset) % num_cities
                travelers.append((home_id, int(traveler_id), dest_id))
        return travelers

    def _execute_trip(self, home_id: int, traveler_id: int, dest_id: int,
                      imported_today: List[int]) -> None:
        """Run one traveler's day trip and apply any transmission.

        The traveler attaches to a random resident "host" in the destination
        and interacts with that host's contacts in the destination's network.
        Transmission is bidirectional and probabilistic per the shared
        ``infection_probability``; the traveler's own state travels with them.

        Args:
            home_id: The traveler's home city id.
            traveler_id: The traveler's id within their home city.
            dest_id: The destination city id.
            imported_today: Per-city running tally of travel-caused exposures,
                mutated in place.
        """
        home = self.cities[home_id]
        dest = self.cities[dest_id]
        p = self.config.infection_probability

        state_before = home.get_individual_state(traveler_id)

        host_id = int(self.rng.integers(0, self.config.population_per_city))
        contacts = dest.contacts_of(host_id, self.rng)

        acquired = False
        residents_infected = 0

        if state_before is State.INFECTIOUS:
            # Infectious visitor may seed the destination's susceptibles.
            for cid in contacts:
                if dest.get_individual_state(int(cid)) is State.SUSCEPTIBLE:
                    if self.rng.random() < p and dest.expose(int(cid)):
                        residents_infected += 1
                        dest.imported_infections += 1
                        imported_today[dest_id] += 1
        elif state_before is State.SUSCEPTIBLE:
            # Susceptible visitor may be exposed by infectious residents and
            # carries that infection home.
            for cid in contacts:
                if dest.get_individual_state(int(cid)) is State.INFECTIOUS:
                    if self.rng.random() < p:
                        acquired = True
                        break
            if acquired and home.expose(traveler_id):
                home.imported_infections += 1
                imported_today[home_id] += 1
        # EXPOSED (incubating) or RECOVERED travelers carry their state home but
        # neither transmit nor are (re)infected during the visit.

        state_after = home.get_individual_state(traveler_id)
        self.travel_events.append(TravelEvent(
            day=self._day,
            traveler_id=traveler_id,
            home_city_id=home_id,
            destination_city_id=dest_id,
            state_before=state_before,
            state_after=state_after,
            acquired_infection=acquired,
            residents_infected=residents_infected,
        ))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def _compute_regional_stats(self, num_travelers: int,
                                new_imported: int) -> Dict:
        """Aggregate current per-city counts into a regional-statistics dict.

        Args:
            num_travelers: Number of travelers that moved today.
            new_imported: Travel-caused exposures created today.

        Returns:
            Dict of regional totals for the current day.
        """
        total_s = total_e = total_i = total_r = 0
        for city in self.cities:
            last = city.history[-1]
            total_s += last.susceptible
            total_e += last.exposed
            total_i += last.infectious
            total_r += last.recovered

        return {
            "day": self._day,
            "total_susceptible": total_s,
            "total_exposed": total_e,
            "total_infectious": total_i,
            "total_recovered": total_r,
            "total_population": total_s + total_e + total_i + total_r,
            "num_travelers": num_travelers,
            "new_imported": new_imported,
            "cumulative_imported": sum(c.imported_infections
                                       for c in self.cities),
        }

    def city_summary(self, city_id: int) -> Dict:
        """Return the epidemic summary for one city.

        Args:
            city_id: The city to summarize.

        Returns:
            The city's :meth:`City.summary_stats` dict.
        """
        if not 0 <= city_id < len(self.cities):
            raise ValueError(f"Invalid city_id: {city_id}")
        return self.cities[city_id].summary_stats()

    def regional_summary(self) -> Dict:
        """Compute overall regional statistics for the completed run.

        Returns:
            Dict with per-city arrival days, imported-infection and travel-event
            counts, and the region-wide attack rate.
        """
        num_cities = len(self.cities)
        city_summaries = [self.city_summary(i) for i in range(num_cities)]

        first_infection_days = [s["first_infection_day"] for s in city_summaries]

        total_infected = sum(s["total_infected"] for s in city_summaries)
        total_pop = sum(s["population"] for s in city_summaries)
        regional_attack_rate = total_infected / total_pop if total_pop else 0.0

        imported_infections = sum(c.imported_infections for c in self.cities)

        return {
            "num_cities": num_cities,
            "total_population": total_pop,
            "total_infected": total_infected,
            "regional_attack_rate": regional_attack_rate,
            "city_first_infection_days": first_infection_days,
            # Arrival day in "City B" (index 1) if it exists, else -1.
            "city_b_first_infection_day": (
                first_infection_days[1] if num_cities > 1 else -1.0),
            "num_travel_events": len(self.travel_events),
            "imported_infections": imported_infections,
            "city_summaries": city_summaries,
        }
