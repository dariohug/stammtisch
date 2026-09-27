"""Use stammtisch agents inside HSLU jass-kit (arena, player service, DL4G tournaments).

    from stammtisch.jasskit_agent import StammtischAgent
    agent = StammtischAgent("pimc")          # any spec: net, pimc:n=64, heuristic, ...

jass-kit plays without Weis/Stöck/multipliers, so the agent uses plain rules.
"""
from __future__ import annotations

import os

import numpy as np
from jass.agents.agent import Agent as JassKitAgent
from jass.game.const import PUSH
from jass.game.game_observation import GameObservation

from . import engine
from .engine import ROOT, lib


def _bits(one_hot: np.ndarray) -> int:
    return sum(1 << int(c) for c in np.flatnonzero(one_hot))


class StammtischAgent(JassKitAgent):
    def __init__(self, spec: str = "pimc", seed: int = 0):
        os.environ.setdefault("STAMMTISCH_MODELS", str(ROOT / "models"))
        self.agent = engine.Agent(spec)
        self.rng = np.random.default_rng(seed)

    def action_trump(self, obs: GameObservation) -> int:
        pushed = obs.forehand == 0
        t = lib.jass_agent_trump(self.agent._p, _bits(obs.hand), int(obs.dealer), int(pushed),
                                 engine.RULES["plain"], int(self.rng.integers(1 << 62)))
        return PUSH if t == 10 else int(t)

    def action_play_card(self, obs: GameObservation) -> int:
        n = int(obs.nr_played_cards)
        played = np.ascontiguousarray(obs.tricks.flatten()[:n], dtype=np.int8)
        buf = np.full(36, -1, np.int8)
        buf[:n] = played
        card = lib.jass_agent_card(self.agent._p, _bits(obs.hand), buf, n, int(obs.player), int(obs.dealer),
                                   int(obs.trump), int(obs.forehand == 0), engine.RULES["plain"],
                                   int(self.rng.integers(1 << 62)))
        if card < 0:
            raise RuntimeError(f"stammtisch could not act on this observation (code {card})")
        return int(card)
