"""The item bank, and item selection.

Selection is where an adaptive test earns its shorter length and where it can
quietly destroy itself. Maximum-information selection always picks the item
closest in difficulty to the current estimate — which means, across a cohort,
the same few hundred items get shown to almost everybody while the rest of the
bank is never used. In a real certification programme that is not an efficiency
note, it is item exposure: the popular items leak, and the test stops measuring.

The fix here is the standard one — randomesque selection: compute the most
informative `k` items and pick one at random. Test length grows by a fraction of
an item; bank usage spreads out enormously. `exposure_rate` in the simulation
report shows what that trade actually costs.
"""
from __future__ import annotations

import dataclasses
import random

from .model import information


@dataclasses.dataclass
class Item:
    id: str
    difficulty: float
    content: str = "general"

    def probability_correct(self, ability: float) -> float:
        from .model import probability
        return probability(ability, self.difficulty)


def build_bank(size: int = 300, seed: int = 11, spread: float = 1.6,
               contents: tuple[str, ...] = ("safety", "tools", "materials", "regulation")) -> list[Item]:
    """A synthetic bank, difficulties normal around 0.

    Real banks are lumpy — plenty of middling items, few at the extremes — and a
    normal spread reproduces that well enough to expose the consequence: the
    tails of the ability scale are always measured least precisely, because
    there is nothing there to measure them with.
    """
    rng = random.Random(seed)
    return [Item(id=f"I{i:04d}",
                 difficulty=round(rng.gauss(0.0, spread), 3),
                 content=contents[i % len(contents)])
            for i in range(size)]


def select(bank: list[Item], ability: float, administered: set[str],
           randomesque: int = 1, rng: random.Random | None = None,
           content: str | None = None) -> Item | None:
    """Most informative item at `ability`, optionally sampled from the top `k`."""
    available = [item for item in bank
                 if item.id not in administered
                 and (content is None or item.content == content)]
    if not available:
        return None
    available.sort(key=lambda item: -information(ability, item.difficulty))
    top = available[:max(1, randomesque)]
    return (rng or random).choice(top) if len(top) > 1 else top[0]
