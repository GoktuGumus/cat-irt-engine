"""One candidate's adaptive test.

The loop is four lines of logic and three policies:

    while not stopped:
        item = most informative item for the current estimate
        response = the candidate answers
        estimate = re-score with the new response

The policies are what make it a test rather than a demo:

* **minimum length** — no result before enough items, however precise the
  estimate looks. Three lucky answers produce a confident wrong number.
* **target standard error** — stop when the measurement is precise enough,
  which is the point of adaptive testing: candidates far from the mean finish
  early, candidates near a decision boundary get more items.
* **maximum length** — nobody sits forever. An ability the bank cannot measure
  precisely must terminate and say so.
"""
from __future__ import annotations

import dataclasses
import random

from .ability import mle, posterior_sd
from .bank import Item, select
from .model import standard_error


@dataclasses.dataclass
class Administered:
    item: Item
    response: int
    ability_after: float
    se_after: float


@dataclasses.dataclass
class Result:
    ability: float
    standard_error: float
    items: list[Administered]
    stopped_by: str

    @property
    def length(self) -> int:
        return len(self.items)

    @property
    def responses(self) -> list[int]:
        return [a.response for a in self.items]


class AdaptiveTest:
    def __init__(self, bank: list[Item], min_items: int = 8, max_items: int = 40,
                 target_se: float = 0.30, randomesque: int = 5,
                 start_ability: float = 0.0, rng: random.Random | None = None):
        self.bank = bank
        self.min_items, self.max_items = min_items, max_items
        self.target_se = target_se
        self.randomesque = randomesque
        self.ability = start_ability
        self.rng = rng or random.Random()

        self.administered: list[Administered] = []
        self._seen: set[str] = set()

    @property
    def difficulties(self) -> list[float]:
        return [a.item.difficulty for a in self.administered]

    @property
    def responses(self) -> list[int]:
        return [a.response for a in self.administered]

    def next_item(self) -> Item | None:
        return select(self.bank, self.ability, self._seen, self.randomesque, self.rng)

    def record(self, item: Item, response: int) -> None:
        self._seen.add(item.id)
        self.ability = mle(self.difficulties + [item.difficulty], self.responses + [response],
                           start=self.ability)
        self.administered.append(
            Administered(item, response, self.ability, self.current_se()))

    def current_se(self) -> float:
        """Posterior SD while the pattern is degenerate, Fisher SE afterwards.

        Reporting 1/√I for an all-correct pattern would claim a precision the
        data does not contain — the estimate is not at that ability, it is merely
        not yet contradicted.
        """
        if not self.administered:
            return float("inf")
        responses = self.responses
        if all(responses) or not any(responses):
            return posterior_sd(self.difficulties, responses)
        return standard_error(self.ability, self.difficulties)

    def stopped(self) -> str | None:
        if len(self.administered) >= self.max_items:
            return "max_items"
        if len(self.administered) < self.min_items:
            return None
        if self.current_se() <= self.target_se:
            return "target_se"
        if not any(item.id not in self._seen for item in self.bank):
            return "bank_exhausted"
        return None

    def run(self, answer) -> Result:
        """`answer(item, ability) -> 0 | 1` supplies the candidate's responses."""
        while True:
            reason = self.stopped()
            if reason:
                return Result(self.ability, self.current_se(), self.administered, reason)
            item = self.next_item()
            if item is None:
                return Result(self.ability, self.current_se(), self.administered,
                              "bank_exhausted")
            self.record(item, answer(item, self.ability))
