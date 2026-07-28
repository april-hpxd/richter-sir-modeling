"""The travel layer: who goes where, for how long.

Model
* Each city has a fixed pool of eligible commuters (``travel_fraction`` of its
  residents), chosen once from the master RNG.
* Each day, every eligible resident who is currently home travels with the
  total probability given by their row of the (possibly asymmetric)
  ``travel_matrix``; the destination is chosen in proportion to that row.
* A trip lasts a number of days drawn from ``trip_duration_distribution``.
  While away the traveler is absent from their home city's local dynamics but
  interacts each day in the destination and ages their own disease; their state
  is mirrored back into the home census daily and written back on return.

Everything is driven from the master generator, so a whole regional run --
cities, trips, destinations, durations, and travel-time transmission -- is
reproducible from one seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.random import Generator

from city import City, DiseaseToken
from config import Config
from disease_model import State


@dataclass
class Traveler:
    """One person currently on a trip.

    Attributes:
        home_city_id: City the traveler departs from and returns to.
        home_individual_id: The traveler's id within their home city.
        current_city_id: City they are visiting.
        days_remaining: Days left before they return home.
        token: Their detached, evolving disease state while away.
        state_before: Disease state at departure (for the trip record).
        residents_infected: Running count of destination residents they have
            exposed during the trip.
        acquired: Whether they were themselves infected during the trip.
    """

    home_city_id: int
    home_individual_id: int
    current_city_id: int
    days_remaining: int
    token: DiseaseToken
    state_before: State
    day_departed: int
    residents_infected: int = 0
    acquired: bool = False


@dataclass
class TravelEvent:
    """Record of one completed trip (kept for statistics and visualization)."""

    day_departed: int
    day_returned: int
    traveler_id: int
    home_city_id: int
    destination_city_id: int
    state_before: State
    state_after: State
    acquired_infection: bool
    residents_infected: int

    # Backwards-compatible alias used by the animation's per-day arrows.
    @property
    def day(self) -> int:
        return self.day_departed


@dataclass(frozen=True)
class InterCityTransmission:
    """A persistent cross-city transmission edge created by travel."""

    day: int
    source_city_id: int
    source_individual_id: int
    target_city_id: int
    target_individual_id: int


@dataclass
class TravelDayResult:
    """What happened, travel-wise, on a single day."""

    imported_today: List[int]
    num_departed: int
    num_active: int
    num_returned: int


class TravelManager:
    """Coordinate temporary, multi-day travel between independent cities."""

    def __init__(self, config: Config, cities: List[City],
                 rng: Generator) -> None:
        """Build the commuter pools and cache the resolved travel structure.

        Args:
            config: Global configuration (travel matrix, durations, fractions).
            cities: The regional cities (referenced, never mutated structurally).
            rng: The master generator; all travel randomness is drawn from it.
        """
        self.config = config
        self.cities = cities
        self.rng = rng

        self.matrix = config.travel_probability_matrix()
        self._durations, self._duration_probs = config.trip_durations()

        # Fixed eligible-commuter pool per city (sorted for deterministic order).
        self.eligible: List[np.ndarray] = []
        for city in cities:
            size = city.config.population_size
            pool_size = int(size * config.travel_fraction)
            if pool_size > 0:
                pool = rng.choice(size, size=pool_size, replace=False)
            else:
                pool = np.empty(0, dtype=np.int64)
            self.eligible.append(np.sort(pool))

        self.active: List[Traveler] = []
        self._away: List[set] = [set() for _ in cities]
        self.travel_events: List[TravelEvent] = []
        self.intercity_transmissions: List[InterCityTransmission] = []

    # 
    def step(self, day: int) -> TravelDayResult:
        """Advance travel by one day: host trips already underway, then depart new ones.

        Order matters for correctness. By the time this runs, every city has
        already called ``advance_disease()`` for today (see
        :meth:`RegionalSimulation.step`), which progresses every currently
        *present* resident -- including someone about to depart, since their
        ``checkout()`` hasn't happened yet. If we then also hosted them at
        their destination today, their disease would advance **twice** on
        their departure day (once at home, once abroad). Hosting existing
        travelers *before* starting new trips means a new departure is
        checked out (capturing today's already-advanced state) but not hosted
        -- and therefore not progressed again -- until tomorrow, giving exactly
        one progression step per person per day in every case.

        Args:
            day: The current simulated day (for event/edge records).

        Returns:
            A :class:`TravelDayResult` with per-city imported exposures and
            trip counts.
        """
        imported_today = [0] * len(self.cities)
        num_returned = self._host_active_trips(day, imported_today)
        num_departed = self._start_new_trips(day)
        return TravelDayResult(
            imported_today=imported_today,
            num_departed=num_departed,
            num_active=len(self.active),
            num_returned=num_returned,
        )

    # 
    def _start_new_trips(self, day: int) -> int:
        """Select today's departures per the travel matrix and check them out."""
        num_cities = len(self.cities)
        if num_cities < 2:
            return 0

        departed = 0
        for home_id, pool in enumerate(self.eligible):
            row = self.matrix[home_id]
            total_p = float(row.sum())
            if total_p <= 0.0 or len(pool) == 0:
                continue

            # Residents eligible AND currently home (not already travelling).
            away = self._away[home_id]
            home_pool = [int(r) for r in pool if int(r) not in away]
            if not home_pool:
                continue

            draws = self.rng.random(len(home_pool))
            dest_probs = row / total_p
            for resident_id, draw in zip(home_pool, draws):
                if draw >= total_p:
                    continue
                dest_id = int(self.rng.choice(num_cities, p=dest_probs))
                duration = int(self.rng.choice(self._durations,
                                               p=self._duration_probs))
                token = self.cities[home_id].checkout(resident_id)
                self.active.append(Traveler(
                    home_city_id=home_id,
                    home_individual_id=resident_id,
                    current_city_id=dest_id,
                    days_remaining=duration,
                    token=token,
                    state_before=token.state,
                    day_departed=day,
                ))
                away.add(resident_id)
                departed += 1
        return departed

    def _host_active_trips(self, day: int,
                           imported_today: List[int]) -> int:
        """Run one day for every active traveler; return those who came home."""
        returned = 0
        still_active: List[Traveler] = []
        for tr in self.active:
            dest = self.cities[tr.current_city_id]
            result = dest.host_visitor_day(tr.token, self.rng)

            # Residents this infectious visitor exposed -> imported to dest.
            for rid in result.infected_resident_ids:
                imported_today[tr.current_city_id] += 1
                dest.imported_infections += 1
                tr.residents_infected += 1
                self.intercity_transmissions.append(InterCityTransmission(
                    day=day,
                    source_city_id=tr.home_city_id,
                    source_individual_id=tr.home_individual_id,
                    target_city_id=tr.current_city_id,
                    target_individual_id=rid,
                ))

            # Visitor newly infected abroad -> imported to their home city.
            if result.acquired_from is not None and not tr.acquired:
                tr.acquired = True
                home = self.cities[tr.home_city_id]
                home.imported_infections += 1
                imported_today[tr.home_city_id] += 1
                self.intercity_transmissions.append(InterCityTransmission(
                    day=day,
                    source_city_id=tr.current_city_id,
                    source_individual_id=result.acquired_from,
                    target_city_id=tr.home_city_id,
                    target_individual_id=tr.home_individual_id,
                ))

            # Mirror the evolving state into the home census (still away)
            self.cities[tr.home_city_id].sync_absent(
                tr.home_individual_id, tr.token)

            tr.days_remaining -= 1
            if tr.days_remaining <= 0:
                self.cities[tr.home_city_id].checkin(
                    tr.home_individual_id, tr.token)
                self._away[tr.home_city_id].discard(tr.home_individual_id)
                self.travel_events.append(TravelEvent(
                    day_departed=tr.day_departed,
                    day_returned=day,
                    traveler_id=tr.home_individual_id,
                    home_city_id=tr.home_city_id,
                    destination_city_id=tr.current_city_id,
                    state_before=tr.state_before,
                    state_after=tr.token.state,
                    acquired_infection=tr.acquired,
                    residents_infected=tr.residents_infected,
                ))
                returned += 1
            else:
                still_active.append(tr)
        self.active = still_active
        return returned
