"""Differential tests against the HSLU reference implementation (jass-kit-py) and the Swisslos logs."""
import numpy as np
import pytest

from stammtisch import engine
from stammtisch.engine import VecEnv, mask_to_cards

jass_rule = pytest.importorskip("jass.game.rule_schieber")


def _collect_states(n_env=256, steps=40 * 12, seed=7):
    """Random-play states (hand, trick, trump) as seen by the player to act."""
    env = VecEnv(n_env, seed=seed)
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(steps):
        mask = env.legal_mask()
        for i in range(n_env):
            trump, n_played = int(env.info[i, 0]), int(env.info[i, 4])
            if trump < 0:
                continue
            k = n_played % 4
            trick = env.history[i, n_played - k:n_played].tolist()
            states.append((int(env.hands[i, env.seat[i]]), trick, trump))
        actions = np.array([rng.choice(np.flatnonzero(m)) for m in mask], np.int16)
        env.step(actions)
    return states


def _classify(hand, trick, trump):
    """True if the trick holds >=2 trumps whose index order differs from their trump rank."""
    trumps = [c for c in trick[1:] if trump < 4 and c // 9 == trump]
    if len(trumps) < 2:
        return False
    strength = [6, 5, 4, 8, 3, 7, 2, 1, 0]
    by_rank = max(trumps, key=lambda c: strength[c % 9])
    by_index = max(trumps)  # jass-kit keeps the larger card index
    return by_rank != by_index


def test_legal_moves_match_jass_kit():
    rule = jass_rule.RuleSchieber()
    states = _collect_states()
    mismatches, explained = 0, 0
    for hand, trick, trump in states:
        hand_1h = np.array([(hand >> c) & 1 for c in range(36)], np.int32)
        ref = rule.get_valid_cards(hand_1h, np.array(trick + [-1] * (4 - len(trick))), len(trick), trump)
        ours = engine.legal(hand, trick, trump)
        if set(np.flatnonzero(ref)) != set(mask_to_cards(ours)):
            mismatches += 1
            explained += _classify(hand, trick, trump)
    print(f"{len(states)} states, {mismatches} mismatches, {explained} explained by jass-kit trump-order bug")
    assert len(states) > 50_000
    # Every disagreement is the known jass-kit issue (picks highest trump by card index, not rank).
    assert mismatches == explained


def test_trick_winner_matches_jass_kit():
    rule = jass_rule.RuleSchieber()
    rng = np.random.default_rng(1)
    for _ in range(20_000):
        trick = rng.choice(36, 4, replace=False)
        trump = int(rng.integers(6))
        ref = (0 - rule.calc_winner(trick, 0, trump)) % 4  # jass-kit returns a seat; seat 0 led
        assert engine.trick_winner_pos(trick.tolist(), trump) == ref


def test_swisslos_logs_replay_cleanly():
    from stammtisch import data
    if not data.PROCESSED.exists():
        pytest.skip("run `make data` first")
    d = data.load()
    status, _ = engine.replay_check(d)
    assert len(status) > 1_000_000
    assert (status == 0).all(), np.bincount(status)


def test_jasskit_adapter_plays_valid_games():
    """Our agent inside jass-kit's own arena: every move is validated by jass-kit."""
    import logging
    from jass.agents.agent_random_schieber import AgentRandomSchieber
    from jass.arena.arena import Arena
    from stammtisch.jasskit_agent import StammtischAgent
    logging.disable(logging.WARNING)
    arena = Arena(nr_games_to_play=40, print_every_x_games=10**9, check_move_validity=True)
    arena.set_players(StammtischAgent("heuristic"), AgentRandomSchieber(), StammtischAgent("heuristic"),
                      AgentRandomSchieber())
    arena.play_all_games()
    assert arena.points_team_0.mean() > arena.points_team_1.mean()
