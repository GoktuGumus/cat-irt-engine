#!/usr/bin/env python3
"""Does the adaptive test actually save anything?

    python -m irt.simulate --candidates 2000

Simulated candidates with known true abilities answer according to the model, so
the estimate can be compared against the truth — which is the only setting where
"how accurate is this test" has a checkable answer.

Two tests are run for every candidate:

* the **adaptive** test, stopping when the standard error is small enough
* a **fixed-length** test of the same bank, which is what it replaces

The comparison that matters is not "which is shorter" — a shorter test is always
available by asking fewer questions. It is *how short the adaptive test can be
while measuring at least as precisely as the fixed one*, which is what the table
at the end reports.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics

from .ability import mle
from .bank import build_bank
from .model import probability, standard_error
from .session import AdaptiveTest


def responder(true_ability: float, rng: random.Random):
    """A candidate who answers exactly as the model says they should."""
    def answer(item, _estimate):
        return int(rng.random() < probability(true_ability, item.difficulty))
    return answer


def fixed_length_test(bank, true_ability: float, length: int, rng: random.Random):
    """The non-adaptive comparison: a fixed set spread across the difficulty range."""
    ordered = sorted(bank, key=lambda item: item.difficulty)
    stride = max(1, len(ordered) // length)
    form = ordered[::stride][:length]
    responses = [int(rng.random() < probability(true_ability, item.difficulty)) for item in form]
    difficulties = [item.difficulty for item in form]
    estimate = mle(difficulties, responses)
    return estimate, standard_error(estimate, difficulties), len(form)


def simulate(candidates: int = 2000, fixed_length: int = 40, target_se: float = 0.40,
             min_items: int = 8, max_items: int = 40, randomesque: int = 5,
             bank_size: int = 300, seed: int = 5) -> dict:
    rng = random.Random(seed)
    bank = build_bank(bank_size, seed=seed + 1)
    exposure = {item.id: 0 for item in bank}

    adaptive_lengths, adaptive_errors, adaptive_ses = [], [], []
    abilities = []
    fixed_errors, fixed_ses = [], []
    stop_reasons: dict[str, int] = {}

    for _ in range(candidates):
        true_ability = rng.gauss(0, 1)

        test = AdaptiveTest(bank, min_items=min_items, max_items=max_items,
                            target_se=target_se, randomesque=randomesque, rng=rng)
        result = test.run(responder(true_ability, rng))
        for administered in result.items:
            exposure[administered.item.id] += 1
        adaptive_lengths.append(result.length)
        adaptive_errors.append(result.ability - true_ability)
        adaptive_ses.append(result.standard_error)
        stop_reasons[result.stopped_by] = stop_reasons.get(result.stopped_by, 0) + 1

        estimate, se, _ = fixed_length_test(bank, true_ability, fixed_length, rng)
        fixed_errors.append(estimate - true_ability)
        fixed_ses.append(se)
        abilities.append(true_ability)

    def rmse(errors):
        return round((sum(e * e for e in errors) / len(errors)) ** 0.5, 3)

    used = sum(1 for count in exposure.values() if count)
    most_used = max(exposure.values()) / candidates

    # The fair comparison is not "which test is shorter" — any test is shorter if
    # you ask fewer questions. It is how long a fixed form has to be to measure
    # as accurately as the adaptive one does.
    adaptive_rmse = (sum(e * e for e in adaptive_errors) / len(adaptive_errors)) ** 0.5
    matched_length, matched_rmse = None, None
    sweep = []
    for length in range(10, 91, 5):
        sweep_rng = random.Random(seed + 99)
        errors = [fixed_length_test(bank, theta, length, sweep_rng)[0] - theta
                  for theta in abilities]
        rmse_at_length = (sum(e * e for e in errors) / len(errors)) ** 0.5
        sweep.append({"length": length, "rmse": round(rmse_at_length, 3)})
        if matched_length is None and rmse_at_length <= adaptive_rmse:
            matched_length, matched_rmse = length, rmse_at_length

    return {
        "candidates": candidates,
        "bank_size": bank_size,
        "target_se": target_se,
        "adaptive": {
            "median_length": statistics.median(adaptive_lengths),
            "mean_length": round(statistics.mean(adaptive_lengths), 1),
            "rmse": rmse(adaptive_errors),
            "bias": round(statistics.mean(adaptive_errors), 3),
            "median_se": round(statistics.median(adaptive_ses), 3),
            "stopped_by": stop_reasons,
        },
        "fixed": {
            "length": fixed_length,
            "rmse": rmse(fixed_errors),
            "bias": round(statistics.mean(fixed_errors), 3),
            "median_se": round(statistics.median(fixed_ses), 3),
        },
        "matched_precision": {
            "fixed_length_needed": matched_length,
            "fixed_rmse": round(matched_rmse, 3) if matched_rmse else None,
            "reduction_pct": (round((1 - statistics.mean(adaptive_lengths) / matched_length) * 100, 1)
                              if matched_length else None),
            "sweep": sweep,
        },
        "bank_usage": {"items_used": used, "of": bank_size,
                       "max_exposure_rate": round(most_used, 3)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="irt.simulate")
    parser.add_argument("--candidates", type=int, default=2000)
    parser.add_argument("--fixed-length", type=int, default=40)
    parser.add_argument("--target-se", type=float, default=0.40)
    parser.add_argument("--randomesque", type=int, default=5)
    parser.add_argument("--json", help="write the full report here")
    args = parser.parse_args()

    report = simulate(candidates=args.candidates, fixed_length=args.fixed_length,
                      target_se=args.target_se, randomesque=args.randomesque)

    adaptive, fixed = report["adaptive"], report["fixed"]
    print(f"\n{report['candidates']} simulated candidates · bank of {report['bank_size']} "
          f"· target SE {report['target_se']}\n")
    print(f"{'':<12} {'items':>7} {'RMSE':>8} {'bias':>8} {'median SE':>11}")
    print("-" * 50)
    print(f"{'adaptive':<12} {adaptive['mean_length']:>7} {adaptive['rmse']:>8} "
          f"{adaptive['bias']:>8} {adaptive['median_se']:>11}")
    print(f"{'fixed':<12} {fixed['length']:>7} {fixed['rmse']:>8} "
          f"{fixed['bias']:>8} {fixed['median_se']:>11}")
    matched = report["matched_precision"]
    if matched["fixed_length_needed"]:
        print(f"\na fixed form needs {matched['fixed_length_needed']} items to reach the same "
              f"accuracy (RMSE {matched['fixed_rmse']}) — "
              f"the adaptive test gets there in {adaptive['mean_length']}, "
              f"{matched['reduction_pct']}% shorter")
    else:
        print("\nno fixed length in the sweep matched the adaptive accuracy")
    print(f"stopped by: {adaptive['stopped_by']}")
    usage = report["bank_usage"]
    print(f"bank usage: {usage['items_used']}/{usage['of']} items, "
          f"most-exposed item seen by {usage['max_exposure_rate']:.1%} of candidates")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
