"""The interaction layer: who meets whom each day.

This is the single seam that the whole architecture is designed around. The
disease engine never decides *how* people come into contact; it only asks a
:class:`ContactModel` "who does person ``i`` interact with today?" and applies
transmission along whatever ids come back.

This milestone ships one implementation, :class:`WellMixedContactModel`, in
which any individual may meet any other (a temporary, homogeneous-mixing
assumption). The explicit goal is that the *next* milestone can add a
``WattsStrogatzContactModel`` -- returning a node's graph neighbours instead of
random strangers -- by writing a new subclass here and changing nothing in
:mod:`engine`, :mod:`disease_model`, or the disease progression itself.

All randomness flows through a single NumPy :class:`~numpy.random.Generator`
passed in by the caller, so contact draws stay part of the one reproducible
random stream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator


class ContactModel(ABC):
    """Abstract interface for a daily contact structure.

    A contact model answers one question: given an individual, which other
    individuals do they interact with on a given day? Concrete subclasses
    define the population structure (well-mixed, networked, spatial, ...)
    without the engine needing to know which.
    """

    @abstractmethod
    def contacts(self, individual_id: int, rng: Generator) -> np.ndarray:
        """Return the ids that ``individual_id`` interacts with today.

        Args:
            individual_id: The id of the (infectious) individual seeking
                contacts.
            rng: The shared random generator; any stochastic contact model must
                draw from this and only this, to preserve reproducibility.

        Returns:
            A 1-D array of *other* individual ids (never including
            ``individual_id`` itself). Ids may be susceptible or not; the
            engine decides what happens on each contact.
        """
        raise NotImplementedError


class WellMixedContactModel(ContactModel):
    """Homogeneous mixing: contacts are uniformly random other individuals.

    Each day an individual meets ``daily_contacts`` distinct other people drawn
    uniformly at random from the whole population. This is the deliberately
    simple, temporary stand-in for a real social network, used here to validate
    the disease dynamics in isolation.

    Attributes:
        population_size: Total number of individuals.
        daily_contacts: Number of distinct others each individual meets daily.
    """

    def __init__(self, population_size: int, daily_contacts: int) -> None:
        """Initialise the well-mixed model.

        Args:
            population_size: Total number of individuals (>= 2).
            daily_contacts: Distinct contacts per individual per day; clamped
                to at most ``population_size - 1`` (you cannot meet more than
                everyone else).
        """
        self.population_size = population_size
        self.daily_contacts = min(daily_contacts, population_size - 1)

    def contacts(self, individual_id: int, rng: Generator) -> np.ndarray:
        """Sample ``daily_contacts`` distinct other individuals uniformly.

        Self-contact is excluded by sampling ``daily_contacts`` positions from
        the ``population_size - 1`` *other* individuals and mapping any position
        at or beyond ``individual_id`` up by one. This is exact (no rejection
        loop) and consumes the random stream deterministically.

        Args:
            individual_id: The individual seeking contacts.
            rng: The shared random generator.

        Returns:
            A 1-D array of distinct other individual ids.
        """
        # Draw distinct positions in the reduced index space [0, n-2].
        positions = rng.choice(
            self.population_size - 1,
            size=self.daily_contacts,
            replace=False,
        )
        # Map positions >= individual_id up by one to skip the individual.
        return np.where(positions >= individual_id, positions + 1, positions)
