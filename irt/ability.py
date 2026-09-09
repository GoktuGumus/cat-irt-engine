"""Estimating ability from a response pattern.

Maximum likelihood is the default, and it has one failure that every real
deployment meets on day one: **an all-correct or all-incorrect pattern has no
maximum**. The likelihood rises monotonically towards ±∞, so the estimate runs
off the scale. A candidate who gets the first three items right is not infinitely
able; they have simply not been measured yet.

So the estimator is MLE with a Bayesian fallback: when the pattern is all one
way — which is exactly when the test is just starting — expected a posteriori
estimation over a standard normal prior gives a finite, sensible number, and the
adaptive algorithm can pick a sensible next item instead of the hardest one in
the bank.
"""
from __future__ import annotations

import math

from .model import MAX_ABILITY, probability


def mle(difficulties, responses, start: float = 0.0,
        tolerance: float = 1e-5, max_iterations: int = 60) -> float:
    """Newton-Raphson on the 1PL log-likelihood.

    The derivative is Σ(u - P) and the second derivative -ΣP(1-P), both cheap and
    both closed form, which is why Rasch scoring runs in microseconds per
    candidate rather than needing an optimiser.
    """
    if not difficulties:
        return 0.0
    if all(responses) or not any(responses):
        return eap(difficulties, responses)

    ability = start
    for _ in range(max_iterations):
        gradient = 0.0
        hessian = 0.0
        for b, u in zip(difficulties, responses):
            p = probability(ability, b)
            gradient += u - p
            hessian -= p * (1 - p)
        if abs(hessian) < 1e-12:
            break
        step = gradient / hessian
        ability -= step
        ability = max(-MAX_ABILITY, min(MAX_ABILITY, ability))
        if abs(step) < tolerance:
            break
    return ability


def eap(difficulties, responses, prior_mean: float = 0.0, prior_sd: float = 1.0,
        points: int = 61) -> float:
    """Expected a posteriori over a normal prior, by quadrature.

    Finite for every response pattern, including the degenerate ones, which is
    what makes it the right fallback rather than a different philosophy.
    """
    lo, hi = prior_mean - 4 * prior_sd, prior_mean + 4 * prior_sd
    step = (hi - lo) / (points - 1)

    numerator = denominator = 0.0
    for i in range(points):
        theta = lo + i * step
        prior = math.exp(-0.5 * ((theta - prior_mean) / prior_sd) ** 2)
        likelihood = 1.0
        for b, u in zip(difficulties, responses):
            p = probability(theta, b)
            likelihood *= p if u else (1 - p)
        weight = prior * likelihood
        numerator += theta * weight
        denominator += weight

    return prior_mean if denominator == 0 else numerator / denominator


def posterior_sd(difficulties, responses, prior_mean: float = 0.0,
                 prior_sd: float = 1.0, points: int = 61) -> float:
    """Spread of the posterior — the honest uncertainty while MLE is undefined."""
    lo, hi = prior_mean - 4 * prior_sd, prior_mean + 4 * prior_sd
    step = (hi - lo) / (points - 1)
    mean = eap(difficulties, responses, prior_mean, prior_sd, points)

    variance = denominator = 0.0
    for i in range(points):
        theta = lo + i * step
        prior = math.exp(-0.5 * ((theta - prior_mean) / prior_sd) ** 2)
        likelihood = 1.0
        for b, u in zip(difficulties, responses):
            p = probability(theta, b)
            likelihood *= p if u else (1 - p)
        weight = prior * likelihood
        variance += ((theta - mean) ** 2) * weight
        denominator += weight

    return prior_sd if denominator == 0 else math.sqrt(variance / denominator)
