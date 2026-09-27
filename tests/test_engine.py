import numpy as np
import pytest

from stammtisch import engine
from stammtisch.engine import card_index as ci, legal, mask_to_cards, trick_winner_pos

H = lambda *names: sum(1 << ci(n) for n in names)  # noqa: E731
names = lambda mask: sorted(engine.card_name(c) for c in mask_to_cards(mask))  # noqa: E731

HEARTS, OBE, UNE = 1, 4, 5


def test_must_follow_suit():
    hand = H("SA", "S6", "D7")
    assert names(legal(hand, [ci("S9")], OBE)) == ["S6", "SA"]


def test_may_trump_instead_of_following():
    hand = H("SA", "H6", "D7")
    assert names(legal(hand, [ci("S9")], HEARTS)) == ["H6", "SA"]


def test_puur_alone_need_not_follow_trump():
    hand = H("HJ", "SA", "D7")
    assert names(legal(hand, [ci("H9")], HEARTS)) == ["D7", "HJ", "SA"]


def test_no_undertrumping():
    # Spades led, H9 (Nell) already trumped in; H6 would under-trump, HJ over-trumps.
    hand = H("H6", "HJ", "D7")
    assert names(legal(hand, [ci("SA"), ci("H9")], HEARTS)) == ["D7", "HJ"]


def test_undertrump_allowed_with_only_trumps():
    hand = H("H6", "H7")
    assert names(legal(hand, [ci("SA"), ci("H9")], HEARTS)) == ["H6", "H7"]


def test_highest_trump_uses_trump_rank_not_card_index():
    # HA (index 9) then H10 (index 13): HA is the highest trump in the trick, so HK
    # would under-trump. jass-kit picks H10 (larger index) and wrongly allows HK.
    hand = H("HK", "D7")
    assert names(legal(hand, [ci("SK"), ci("HA"), ci("H10")], HEARTS)) == ["D7"]


@pytest.mark.parametrize("trick,trump,winner", [
    (["HA", "H9", "H10", "D10"], HEARTS, 1),   # Nell beats the ace
    (["SA", "S6", "H6", "SK"], HEARTS, 2),     # smallest trump wins
    (["SA", "SK", "S6", "D6"], OBE, 0),
    (["SA", "SK", "S6", "D6"], UNE, 2),
    (["S7", "HJ", "H9", "HA"], HEARTS, 1),
])
def test_trick_winner(trick, trump, winner):
    assert trick_winner_pos([ci(c) for c in trick], trump) == winner


def test_round_points_total_157_or_match():
    env = engine.VecEnv(64, seed=3, rules="plain")
    scores = []
    while len(scores) < 64:
        env.step(env.builtin_actions("random"))
        scores += [s for s, d in zip(env.final_score.tolist(), env.done) if d]
    for a, b in scores:
        assert a + b in (157, 257)


def test_vecenv_rejects_illegal_and_plays_full_rounds():
    env = engine.VecEnv(128, seed=1)
    assert env.step(np.full(128, 99, np.int16)) == 128  # all illegal, nothing changes
    finished = 0
    for _ in range(40 * 10):
        mask = env.legal_mask()
        assert mask.any(1).all()
        rng = np.random.default_rng(0)
        actions = np.array([rng.choice(np.flatnonzero(m)) for m in mask], np.int16)
        assert env.step(actions) == 0
        finished += int(env.done.sum())
    assert finished >= 128
