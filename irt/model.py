"""The Rasch (1PL) model.

One parameter per item: its difficulty `b`, on the same scale as the candidate's
ability `θ`. A candidate of ability θ answers an item of difficulty b correctly
with probability

    P(θ) = 1 / (1 + exp(-(θ - b)))

so θ = b means a coin flip, and every unit of ability is worth the same
everywhere on the scale. That last property is why Rasch is used for
certification: two candidates who answered different items can still be compared,
which is the entire point of adaptive testing.

Fisher information for a 1PL item is P(1-P), maximised at P = 0.5. An item is
most informative when the candidate has an even chance on it — which is also the
reason a well-run adaptive test feels uncomfortably hard to everyone.
"""
from __future__ import annotations

import math

MAX_ABILITY = 4.0        # the scale is standardised; beyond ±4 nothing is measurable


def probability(ability: float, difficulty: float) -> float:
    """P(correct). Guarded against overflow at the tails of the logistic."""
    z = ability - difficulty
    if z > 35:
        return 1.0
    if z < -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def information(ability: float, difficulty: float) -> float:
    """Fisher information contributed by one item at this ability."""
    p = probability(ability, difficulty)
    return p * (1.0 - p)


def test_information(ability: float, difficulties) -> float:
    return sum(information(ability, b) for b in difficulties)


def standard_error(ability: float, difficulties) -> float:
    """SE(θ) = 1/√I. Infinite when nothing informative has been administered."""
    info = test_information(ability, difficulties)
    return float("inf") if info <= 0 else 1.0 / math.sqrt(info)


def log_likelihood(ability: float, difficulties, responses) -> float:
    total = 0.0
    for b, u in zip(difficulties, responses):
        p = min(max(probability(ability, b), 1e-12), 1 - 1e-12)
        total += math.log(p) if u else math.log(1 - p)
    return total
