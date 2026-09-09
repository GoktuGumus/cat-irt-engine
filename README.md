# Computerised Adaptive Testing — Rasch engine

[![tests](https://github.com/GoktuGumus/cat-irt-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/GoktuGumus/cat-irt-engine/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)
![licence](https://img.shields.io/badge/licence-MIT-green)

A working adaptive test: pick the most informative item for what you currently
believe about a candidate, ask it, re-score, stop when the measurement is precise
enough. Rasch (1PL) model, maximum-likelihood scoring with a Bayesian fallback,
maximum-information selection with exposure control, and a simulation that
checks the whole thing against candidates whose true ability is known.

```
                 ┌──────────────────────────────────────────┐
                 ▼                                          │
   ability estimate ──▶ most informative unseen item ──▶ response ──▶ re-score
                 │                                                        │
                 └────────── stop when SE ≤ target, or max items ─────────┘
```

## Why adaptive testing exists

A fixed exam asks everyone the same questions, so most of them are wasted:
easy items tell you nothing about a strong candidate, hard ones tell you nothing
about a weak one. An adaptive test spends its questions where they carry
information — near the candidate's own level — and stops when it knows enough.

Measured over 5,000 simulated candidates (`python -m irt.simulate`):

| | items | RMSE | bias | median SE |
|---|---|---|---|---|
| **adaptive**, stop at SE ≤ 0.40 | **27.3** | 0.401 | −0.001 | 0.396 |
| fixed form, 40 items | 40 | 0.405 | −0.010 | 0.386 |

The honest comparison is not against the 40-item form — it is against however
long a fixed form has to be to measure *as accurately*. That length is **45
items**, so the adaptive test reaches the same accuracy in **39% fewer
questions**, and the saving comes out of the questions that were telling nobody
anything.

Full numbers, including the fixed-length sweep, in
[`docs/simulation.json`](docs/simulation.json).

## The floor nothing can go under

A 1PL item contributes at most **0.25** Fisher information, exactly when the
candidate has a 50/50 chance on it. Since SE = 1/√I, no selection algorithm,
however clever, reaches SE 0.30 in fewer than about 44 items:

```
1 / (0.25 × 0.30²) ≈ 44
```

That is worth stating plainly, because adaptive testing is often sold as though
it makes tests arbitrarily short. It does not. It gets you *to the bound*, where
a badly targeted fixed form wastes most of its items well short of it. The first
simulation run for this repository asked for SE 0.30 in at most 40 items and
every single candidate hit the item limit — the algorithm was fine, the target
was arithmetically impossible. `test_information_bound_sets_the_floor_on_test_length`
now pins that down.

## Scoring, and the failure every deployment meets

Maximum likelihood by Newton-Raphson — the gradient is `Σ(u − P)` and the
Hessian `−ΣP(1−P)`, both closed form, so scoring is microseconds per candidate.

But **an all-correct or all-incorrect pattern has no maximum**. The likelihood
climbs forever and the estimate runs off the scale. This is not an edge case; it
is the first three items of most tests. A candidate who has answered everything
correctly so far is not infinitely able, they are simply not yet measured.

So the estimator falls back to expected a posteriori over a standard normal
prior whenever the pattern is degenerate. Finite, sensible, and it lets item
selection pick a reasonable next question instead of the hardest item in the
bank.

That fallback has a consequence the repository states rather than hides: the two
estimators disagree slightly, so a candidate with one correct answer can score
marginally *lower* than one with none. It happens only where the estimate
carries almost no information, and `min_items` stops such a number ever being
reported — but it is real, and
`test_the_estimator_is_discontinuous_at_the_degenerate_boundary` asserts it so
nobody rediscovers it in production.

The reported standard error follows the same rule: posterior spread while the
pattern is degenerate, Fisher 1/√I once it is not. Quoting 1/√I for an
all-correct pattern would claim a precision the responses do not contain.

## Item selection and exposure

Maximum-information selection always wants the item closest in difficulty to the
current estimate — which means, across a cohort, a few hundred items get shown
to almost everybody and the rest of the bank is never used. In certification that
is not an efficiency footnote: the popular items leak, and the test stops
measuring anything.

Selection is therefore *randomesque*: compute the best `k` items, pick one at
random. In the simulation above, with `k = 5`, all 300 items in the bank are used
and the most-exposed item is seen by 33% of candidates, at a cost of a fraction
of an item in test length.

`select()` also takes a content area, which is how content-balancing constraints
attach without the selection logic knowing anything about them.

## Stopping

Three rules, each preventing a different bad outcome:

| Rule | What it prevents |
|---|---|
| minimum length | A confident result from three lucky answers |
| target standard error | Asking questions after the measurement is good enough |
| maximum length | An unmeasurable candidate sitting forever |

The result records which rule fired. `stopped_by: max_items` across a whole
cohort means the target is unreachable with that bank — which is exactly how the
impossible SE 0.30 target above announced itself.

## Running it

```bash
python -m irt.simulate --candidates 5000 --target-se 0.40
python -m irt.simulate --randomesque 1        # watch bank usage collapse
python -m pytest -q                            # 34 tests
```

Driving a single test from your own code:

```python
from irt.bank import build_bank
from irt.session import AdaptiveTest

test = AdaptiveTest(build_bank(300), min_items=8, max_items=40, target_se=0.4)
while not test.stopped():
    item = test.next_item()
    test.record(item, ask_the_candidate(item))     # 1 correct, 0 incorrect

print(test.ability, test.current_se())
```

Pure standard library — no numpy, no scipy, no dependencies at all.

## Testing

Ability estimation is testable in a way most machine learning is not: choose a
true ability, simulate responses from it, and check the estimate comes back. So
the tests assert recovery, not merely that the code runs.

| What is asserted | Why it matters |
|---|---|
| Five known abilities are recovered within 0.2 | The estimator does its job |
| Information peaks at 0.25 where item matches candidate | The bound that sets minimum test length |
| All-correct and all-incorrect stay on the scale | The failure every deployment meets |
| The estimator's discontinuity at that boundary | Stated rather than discovered later |
| Randomesque selection spreads the bank; greedy does not | Item exposure |
| No item is administered twice; min/max/SE stopping rules hold | Test integrity |
| Adaptive is ≥20% shorter at matched accuracy, and unbiased | The claim the repository makes |

## What this is, and what it isn't

It **is** a complete Rasch adaptive testing engine with an honest evaluation.

It **is not** a full assessment platform: there is no item bank calibration
(difficulties are given, not estimated from response data), no 2PL or 3PL, no
content-balancing constraint solver beyond a single filter, and no candidate
delivery layer. The item bank is generated. No real exam, item or candidate
appears anywhere in this repository.

## Licence

MIT.
