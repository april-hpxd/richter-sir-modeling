"""Regional simulation: many independent cities coupled by a travel layer.

:class:`RegionalSimulation` is a thin coordinator. It owns a list of fully
independent :class:`~city.City` objects (each with its own population, network,
engine, RNG, history, and statistics) and a single :class:`~travel.TravelManager`
that moves people between them. It contains **no** disease logic and **no**
travel logic of its own -- it only sequences the daily update and aggregates
statistics.

Everything is data-driven from :class:`~config.Config`: the number of cities,
each city's population, and the (possibly asymmetric) travel matrix all come
from configuration, so 2, 3, 5, 10, or 20 cities of differing sizes work with
no code change.

Daily update
------------
1. Advance disease **inside** every city (its own network only).
2. Run the travel layer: start new multi-day trips, and for every traveller
   currently away run one day in their destination (bidirectional, network-based
   transmission) and age their disease.
3. Record each city's day, folding travel-imported exposures into that day's
   ``new_exposed`` so imported cases appear in the same day's stats and frames.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from city import City, CityConfig
from config import Config
from travel import InterCityTransmission, TravelEvent, TravelManager


class RegionalSimulation:
    """Coordinate any number of independent cities coupled by travel.

    Attributes:
        config: The global configuration.
        cities: Independent :class:`~city.City` objects, indexed by city id.
        rng: The master generator seeding every city and driving all travel.
        travel: The :class:`~travel.TravelManager`.
        history: One regional-statistics dict per day (including day 0).
    """

    def __init__(self, config: Config) -> None:
        """Build the cities from configuration, seed city 0, and record day 0.

        Args:
            config: Global configuration. ``config.city_sizes()`` determines the
                number of cities and each one's population.
        """
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

        sizes = config.city_sizes()
        self.cities: List[City] = []
        for city_id, size in enumerate(sizes):
            city_rng = np.random.default_rng(int(self.rng.integers(0, 2**31)))
            city_config = CityConfig(
                population_size=size,
                daily_contacts=min(config.daily_contacts, size - 1),
                infection_probability=config.infection_probability,
                incubation_days=config.incubation_days,
                infectious_days=config.infectious_days,
                contact_model_type=config.contact_model,
                watts_strogatz_k=config.watts_strogatz_k,
                watts_strogatz_p=config.watts_strogatz_p,
                random_degree_min=config.random_degree_min,
                random_degree_max=config.random_degree_max,
                behavioral_response_factor=(
                    config.behavioral_response_factor
                    if config.behavioral_response_enabled else None),
                isolation_contact_multiplier=config.isolation_contact_multiplier,
            )
            self.cities.append(City(city_id, city_config, city_rng))

        # The travel layer consumes the master stream *after* city RNGs are
        # drawn, keeping city construction reproducible independently of travel.
        self.travel = TravelManager(config, self.cities, self.rng)

        self.history: List[Dict] = []
        self._day = 0

        # Seed the outbreak in city 0 only; every other city starts fully
        # susceptible so its arrival day is meaningful.
        seeded = config.initial_infected
        self.cities[0].seed_infection(seeded)
        for city in self.cities:
            city.record_day(
                new_exposed=(seeded if city.id == 0 else 0),
                new_infectious=0, new_recovered=0)
        self.history.append(self._compute_regional_stats(
            num_travelers=0, new_imported=0, regional_new=seeded))

    # Convenience pass-throughs so downstream code (visualization, stats) does
    # not need to reach into the travel manager.
    @property
    def travel_events(self) -> List[TravelEvent]:
        return self.travel.travel_events

    @property
    def intercity_transmissions(self) -> List[InterCityTransmission]:
        return self.travel.intercity_transmissions

    # ------------------------------------------------------------------
    def step(self) -> Dict:
        """Advance the whole region by one day and record regional statistics."""
        self._day += 1

        # 1. Internal disease dynamics for every city (not yet recorded).
        deltas = [city.advance_disease() for city in self.cities]

        # 2. Travel: start new trips and host everyone currently away.
        travel_result = self.travel.step(self._day)
        imported = travel_result.imported_today
        locations = self.travel.current_locations()

        # 3. Record each city with internal + imported new exposures.
        regional_new = 0
        for city, delta, imp, location in zip(
                self.cities, deltas, imported, locations):
            new_exposed = delta["new_exposed"] + imp
            regional_new += new_exposed
            city.record_day(
                new_exposed=new_exposed,
                new_infectious=delta["new_infectious"],
                new_recovered=delta["new_recovered"],
                transmissions=delta.get("transmissions"),
                traveler_locations=location)

        # 4. Optional city isolation: toggle once a city's infectious share
        # crosses the configured threshold (one-way -- stays isolated).
        if self.config.isolation_enabled:
            for city in self.cities:
                if city.isolated:
                    continue
                last = city.history[-1]
                prevalence = last.infectious / max(1, city.config.population_size)
                if prevalence >= self.config.isolation_infectious_threshold:
                    city.set_isolated(True)
                    self.travel.set_city_isolated(city.id, True)

        stats = self._compute_regional_stats(
            num_travelers=travel_result.num_departed,
            new_imported=sum(imported), regional_new=regional_new,
            num_active=travel_result.num_active)
        self.history.append(stats)
        return stats

    def has_active_disease_or_travel(self) -> bool:
        """Return ``True`` while any city still has active disease or a traveler remains in transit."""
        return any(city.is_epidemic_active() for city in self.cities) or bool(self.travel.active)

    def run(self, verbose: bool = False) -> List[Dict]:
        """Run to completion (no active disease or travelers left, or day cap)."""
        for _ in range(self.config.simulation_days):
            stats = self.step()
            if verbose:
                print(
                    f"Day {stats['day']:3d} | "
                    f"S={stats['total_susceptible']:5d} "
                    f"E={stats['total_exposed']:4d} "
                    f"I={stats['total_infectious']:4d} "
                    f"R={stats['total_recovered']:5d} | "
                    f"depart={stats['num_travelers']:3d} "
                    f"away={stats['num_active']:3d} "
                    f"imported={stats['new_imported']}")
            if not self.has_active_disease_or_travel():
                break
        return self.history

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def _compute_regional_stats(self, num_travelers: int, new_imported: int,
                                regional_new: int,
                                num_active: int = 0) -> Dict:
        """Aggregate current per-city census counts into a regional dict."""
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
            "regional_new_infections": regional_new,
            "num_travelers": num_travelers,
            "num_active": num_active,
            "new_imported": new_imported,
            "cumulative_imported": sum(c.imported_infections
                                       for c in self.cities),
            "cities_isolated": [c.isolated for c in self.cities],
        }

    def exported_infections(self) -> List[int]:
        """Return, per city, how many other cities' cases it seeded via travel.

        This is the mirror image of ``City.imported_infections``: a count of
        cross-city transmissions whose *source* was a resident or traveler of
        this city. Computed here (not on :class:`~city.City`) because a city
        must not know about other cities -- only the coordinator sees the full
        set of cross-city links.
        """
        counts = [0] * len(self.cities)
        for link in self.intercity_transmissions:
            counts[link.source_city_id] += 1
        return counts

    def city_summary(self, city_id: int) -> Dict:
        """Return the epidemic summary for one city, plus exported_infections."""
        if not 0 <= city_id < len(self.cities):
            raise ValueError(f"Invalid city_id: {city_id}")
        stats = dict(self.cities[city_id].summary_stats())
        stats["exported_infections"] = float(
            self.exported_infections()[city_id])
        return stats

    def infection_sources(self) -> List[int]:
        """Return, per city, the id of the city its infection first arrived from.

        The seed city (and any city never infected) has source ``-1``. For every
        other infected city it is the source city of the earliest cross-city
        transmission targeting it.
        """
        sources = [-1] * len(self.cities)
        earliest: Dict[int, int] = {}
        for link in sorted(self.intercity_transmissions, key=lambda t: t.day):
            tgt = link.target_city_id
            if tgt not in earliest:
                earliest[tgt] = link.day
                sources[tgt] = link.source_city_id
        return sources

    def effective_reproduction_by_generation(self) -> Dict[int, float]:
        """Estimate the effective reproduction number per transmission generation.

        Every individual across every city carries an ``infection_generation``
        (seeded cases are generation 0; each new case is one more than its
        source's generation -- see :attr:`disease_model.Individual`). The
        generation-based estimator ``Rt(g) = count(g + 1) / count(g)`` is the
        mean number of secondary cases produced per case in generation ``g``.

        Returns:
            A dict mapping generation ``g`` to its ``Rt`` estimate, for every
            generation that produced at least one further case (an empty
            generation g+1 gives ``Rt(g) == 0.0``; a generation with no cases
            at all is omitted).
        """
        from disease_model import State

        counts: Dict[int, int] = {}
        for city in self.cities:
            for ind in city.engine.individuals:
                if ind.state is State.SUSCEPTIBLE:
                    continue
                counts[ind.infection_generation] = (
                    counts.get(ind.infection_generation, 0) + 1)

        if not counts:
            return {}
        max_gen = max(counts)
        return {
            g: counts.get(g + 1, 0) / counts[g]
            for g in range(max_gen)
            if counts.get(g, 0) > 0
        }

    def regional_summary(self) -> Dict:
        """Compute overall regional statistics for the completed run."""
        num_cities = len(self.cities)
        city_summaries = [self.city_summary(i) for i in range(num_cities)]
        first_infection_days = [s["first_infection_day"] for s in city_summaries]

        total_infected = sum(s["total_infected"] for s in city_summaries)
        total_pop = sum(s["population"] for s in city_summaries)
        regional_attack_rate = total_infected / total_pop if total_pop else 0.0
        imported_infections = sum(c.imported_infections for c in self.cities)

        # Average arrival delay: mean first-infection day of the non-seed cities
        # that were reached at all (the seed city's day 0 is excluded).
        arrivals = [d for i, d in enumerate(first_infection_days)
                    if i != 0 and d >= 0]
        avg_delay = float(np.mean(arrivals)) if arrivals else -1.0

        rt_by_generation = self.effective_reproduction_by_generation()
        mean_effective_r = (float(np.mean(list(rt_by_generation.values())))
                           if rt_by_generation else 0.0)

        return {
            "num_cities": num_cities,
            "total_population": total_pop,
            "total_infected": total_infected,
            "total_regional_infections": total_infected,
            "regional_attack_rate": regional_attack_rate,
            "city_first_infection_days": first_infection_days,
            "city_b_first_infection_day": (
                first_infection_days[1] if num_cities > 1 else -1.0),
            "infection_sources": self.infection_sources(),
            "average_arrival_delay": avg_delay,
            "cities_reached": sum(1 for d in first_infection_days if d >= 0),
            "daily_regional_infections": [h["regional_new_infections"]
                                          for h in self.history],
            "num_travel_events": len(self.travel_events),
            "imported_infections": imported_infections,
            "city_summaries": city_summaries,
            "effective_r_by_generation": rt_by_generation,
            "mean_effective_r": mean_effective_r,
            "cities_isolated": [c.isolated for c in self.cities],
        }
