"""The model, the estimator, selection, and the test that runs them.

Ability estimation is testable in a way most machine learning is not: simulate a
candidate whose true ability you chose, and check the estimate comes back. So
these tests assert recovery, not just that the code runs.
"""
import random
import statistics

import pytest

from irt.ability import eap, mle, posterior_sd
from irt.bank import Item, build_bank, select
from irt.model import (MAX_ABILITY, information, probability, standard_error,
                       test_information as total_information)
from irt.session import AdaptiveTest
from irt.simulate import responder, simulate


# --------------------------------------------------------------------------- the model

def test_equal_ability_and_difficulty_is_a_coin_flip():
    assert probability(0.5, 0.5) == pytest.approx(0.5)


def test_probability_rises_with_ability():
    values = [probability(theta, 0.0) for theta in (-3, -1, 0, 1, 3)]
    assert values == sorted(values)
    assert 0.0 <= values[0] and values[-1] <= 1.0


def test_probability_survives_the_tails():
    assert probability(500, 0) == 1.0
    assert probability(-500, 0) == 0.0


def test_information_peaks_where_the_item_matches_the_candidate():
    peak = information(0.0, 0.0)
    assert peak == pytest.approx(0.25)
    assert information(0.0, 2.0) < peak
    assert information(0.0, -2.0) < peak


def test_information_bound_sets_the_floor_on_test_length():
    """No 1PL item can contribute more than 0.25, whatever the selection does."""
    best = max(information(0.0, b / 10) for b in range(-40, 41))
    assert best <= 0.25 + 1e-9
    # so SE 0.3 needs at least 1/(0.25 * 0.3^2) items — about 44
    assert standard_error(0.0, [0.0] * 44) < 0.31
    assert standard_error(0.0, [0.0] * 40) > 0.30


def test_standard_error_falls_as_items_accumulate():
    errors = [standard_error(0.0, [0.0] * n) for n in (5, 10, 20, 40)]
    assert errors == sorted(errors, reverse=True)


def test_no_items_means_no_information():
    assert total_information(0.0, []) == 0
    assert standard_error(0.0, []) == float("inf")


# --------------------------------------------------------------------------- estimation

@pytest.mark.parametrize("true_ability", [-1.5, -0.5, 0.0, 0.8, 1.7])
def test_ability_is_recovered_from_simulated_responses(true_ability):
    """60 well-spread items should locate a candidate within a few tenths."""
    rng = random.Random(4)
    difficulties = [round(-3 + 6 * i / 59, 3) for i in range(60)]
    estimates = []
    for _ in range(40):
        responses = [int(rng.random() < probability(true_ability, b)) for b in difficulties]
        estimates.append(mle(difficulties, responses))
    assert statistics.mean(estimates) == pytest.approx(true_ability, abs=0.2)


def test_more_correct_answers_never_lowers_the_estimate():
    """Monotone within the mixed-pattern regime, where MLE is defined."""
    difficulties = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    previous = -99.0
    for correct in range(1, len(difficulties)):
        responses = [1] * correct + [0] * (len(difficulties) - correct)
        estimate = mle(difficulties, responses)
        assert estimate > previous
        previous = estimate


def test_the_estimator_is_discontinuous_at_the_degenerate_boundary():
    """A known property, asserted so it cannot surprise anyone later.

    All-wrong is scored by EAP, which pulls towards the prior; one-correct is
    scored by MLE, which does not. So the second candidate can come out *lower*
    than the first. It only happens where the estimate carries almost no
    information anyway, and `min_items` keeps such a number from being reported —
    but it is real, and pretending otherwise is how it gets discovered in
    production.
    """
    difficulties = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    all_wrong = mle(difficulties, [0] * 6)
    one_right = mle(difficulties, [1] + [0] * 5)
    assert one_right < all_wrong


def test_all_correct_does_not_run_off_the_scale():
    """MLE has no maximum here; the estimate must still be finite and sane."""
    estimate = mle([-1.0, 0.0, 1.0], [1, 1, 1])
    assert 0 < estimate < MAX_ABILITY


def test_all_incorrect_does_not_run_off_the_scale():
    estimate = mle([-1.0, 0.0, 1.0], [0, 0, 0])
    assert -MAX_ABILITY < estimate < 0


def test_eap_is_pulled_towards_the_prior_by_thin_evidence():
    """One correct answer is not evidence of high ability, and EAP says so."""
    thin = eap([0.0], [1])
    thick = eap([0.0] * 20, [1] * 20)
    assert 0 < thin < thick


def test_posterior_spread_shrinks_with_evidence():
    assert posterior_sd([0.0], [1]) > posterior_sd([0.0] * 20, [1] * 20)


def test_no_responses_returns_the_prior_mean():
    assert mle([], []) == 0.0


# --------------------------------------------------------------------------- selection

def test_selection_picks_the_closest_item_to_the_estimate():
    bank = [Item("a", -2.0), Item("b", 0.1), Item("c", 2.0)]
    assert select(bank, 0.0, set()).id == "b"


def test_selection_skips_what_has_already_been_administered():
    bank = [Item("a", -2.0), Item("b", 0.1), Item("c", 2.0)]
    assert select(bank, 0.0, {"b"}).id in {"a", "c"}


def test_an_exhausted_bank_returns_nothing():
    assert select([Item("a", 0.0)], 0.0, {"a"}) is None


def test_randomesque_spreads_the_load_across_similar_items():
    """Always taking the single best item shows the same few to everyone."""
    bank = build_bank(60, seed=2)
    rng = random.Random(1)
    greedy = {select(bank, 0.0, set(), randomesque=1, rng=rng).id for _ in range(30)}
    spread = {select(bank, 0.0, set(), randomesque=5, rng=rng).id for _ in range(30)}
    assert len(greedy) == 1
    assert len(spread) > 1


def test_content_filtering_restricts_the_pool():
    bank = build_bank(40, seed=3)
    chosen = select(bank, 0.0, set(), content="safety")
    assert chosen.content == "safety"


# --------------------------------------------------------------------------- the session

def make_test(**kwargs):
    defaults = dict(bank=build_bank(200, seed=7), rng=random.Random(9))
    return AdaptiveTest(**{**defaults, **kwargs})


def test_the_minimum_length_is_respected_even_when_precision_arrives_early():
    test = make_test(min_items=12, max_items=40, target_se=5.0)   # trivially reachable
    result = test.run(responder(0.0, random.Random(2)))
    assert result.length >= 12


def test_the_maximum_length_is_respected_when_precision_never_arrives():
    test = make_test(min_items=5, max_items=20, target_se=0.01)   # unreachable
    result = test.run(responder(0.0, random.Random(2)))
    assert result.length == 20
    assert result.stopped_by == "max_items"


def test_stopping_at_the_target_precision():
    test = make_test(min_items=8, max_items=60, target_se=0.45)
    result = test.run(responder(0.3, random.Random(5)))
    assert result.stopped_by == "target_se"
    assert result.standard_error <= 0.45


def test_no_item_is_ever_administered_twice():
    test = make_test(min_items=10, max_items=40, target_se=0.3)
    result = test.run(responder(1.0, random.Random(6)))
    ids = [a.item.id for a in result.items]
    assert len(ids) == len(set(ids))


def test_the_estimate_is_recorded_after_every_item():
    test = make_test(min_items=6, max_items=15, target_se=0.35)
    result = test.run(responder(-0.7, random.Random(8)))
    assert all(abs(a.ability_after) <= MAX_ABILITY for a in result.items)
    assert result.items[-1].ability_after == result.ability


def test_an_exhausted_bank_stops_the_test():
    small = AdaptiveTest(build_bank(6, seed=1), min_items=2, max_items=50, target_se=0.01,
                         rng=random.Random(1))
    result = small.run(responder(0.0, random.Random(1)))
    assert result.stopped_by in {"bank_exhausted", "max_items"}
    assert result.length <= 6


# --------------------------------------------------------------------------- the simulation

@pytest.fixture(scope="module")
def report():
    return simulate(candidates=300, fixed_length=40, target_se=0.40, seed=5)


def test_the_adaptive_test_is_shorter_at_matched_accuracy(report):
    matched = report["matched_precision"]
    assert matched["fixed_length_needed"] > report["adaptive"]["mean_length"]
    assert matched["reduction_pct"] > 20


def test_estimates_are_not_systematically_high_or_low(report):
    assert abs(report["adaptive"]["bias"]) < 0.1


def test_a_longer_fixed_form_is_a_more_accurate_one(report):
    """Sanity check on the comparison itself, not on the engine."""
    sweep = report["matched_precision"]["sweep"]
    assert sweep[0]["rmse"] > sweep[-1]["rmse"]


def test_the_bank_is_not_worn_out_by_a_handful_of_items(report):
    usage = report["bank_usage"]
    assert usage["items_used"] > usage["of"] * 0.5
    assert usage["max_exposure_rate"] < 0.6
