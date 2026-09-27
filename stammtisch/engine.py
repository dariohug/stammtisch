"""ctypes binding to the C++ engine (build/libjass.so)."""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
N_ACTIONS = 43  # 0..35 card, 36..41 trump, 42 push
ACT_TRUMP, ACT_PUSH = 36, 42

SUITS = "DHSC"
RANKS = ["A", "K", "Q", "J", "10", "9", "8", "7", "6"]
# Suit names follow Swisslos: Ecken/Eichel ♦ and Herz/Rosen ♥ count x1, Schaufel/Schilten ♠
# and Kreuz/Schellen ♣ count x2, Obenabe/Undeufe x3.
TRUMP_NAMES = ["Ecken/Eichel ♦", "Herz/Rosen ♥", "Schaufel/Schilten ♠", "Kreuz/Schellen ♣", "Obenabe", "Undeufe"]
RULES = {"swisslos": 0, "plain": 1}


def card_index(name: str) -> int:
    return SUITS.index(name[0]) * 9 + RANKS.index(name[1:])


def card_name(idx: int) -> str:
    return SUITS[idx // 9] + RANKS[idx % 9]


def mask_to_cards(mask: int) -> list[int]:
    return [i for i in range(64) if mask >> i & 1]


def _load() -> C.CDLL:
    path = Path(os.environ.get("STAMMTISCH_LIB", ROOT / "build" / "libjass.so"))
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `make` first")
    lib = C.CDLL(str(path))
    i8p, i16p, i32p, u64p = (np.ctypeslib.ndpointer(t, flags="C_CONTIGUOUS")
                             for t in (np.int8, np.int16, np.int32, np.uint64))
    lib.jass_abi_version.restype = C.c_int
    lib.jass_legal.argtypes = [C.c_uint64, i32p, C.c_int32, C.c_int32]
    lib.jass_legal.restype = C.c_uint64
    lib.jass_trick_winner_pos.argtypes = [i32p, C.c_int32]
    lib.jass_trick_winner_pos.restype = C.c_int32
    lib.jass_replay_check.argtypes = [C.c_int64, i8p, i8p, i8p, i16p, i8p, i8p, i8p, i8p]
    lib.jass_replay_check.restype = C.c_int64
    lib.jass_env_create.argtypes = [C.c_int32, C.c_uint64]
    lib.jass_env_create.restype = C.c_void_p
    lib.jass_env_destroy.argtypes = [C.c_void_p]
    lib.jass_env_reset.argtypes = [C.c_void_p]
    lib.jass_env_observe.argtypes = [C.c_void_p, i8p, u64p, u64p, i8p, i16p]
    lib.jass_env_step.argtypes = [C.c_void_p, i16p, C.c_int32, i8p, i16p]
    lib.jass_env_step.restype = C.c_int32
    lib.jass_env_set_rules.argtypes = [C.c_void_p, C.c_int32]
    lib.jass_agent_create.argtypes = [C.c_char_p]
    lib.jass_agent_create.restype = C.c_void_p
    lib.jass_agent_destroy.argtypes = [C.c_void_p]
    lib.jass_env_agent_actions.argtypes = [C.c_void_p, C.c_void_p, i8p, i16p]
    f32p, f64p = (np.ctypeslib.ndpointer(t, flags="C_CONTIGUOUS") for t in (np.float32, np.float64))
    lib.jass_analyze_round.argtypes = [i8p, C.c_int32, C.c_void_p, C.c_int32, C.c_int32, C.c_int32, C.c_int32,
                                       C.c_int32, C.c_int32, C.c_int32, C.c_void_p, C.c_double, C.c_uint64, u64p,
                                       f32p, f32p, f32p]
    lib.jass_analyze_round.restype = C.c_int32
    lib.jass_analyze_trump.argtypes = [C.c_uint64, C.c_void_p, C.c_int32, C.c_int32, C.c_int32, C.c_int32,
                                       C.c_int32, C.c_void_p, C.c_double, C.c_uint64, f64p, f64p]
    lib.jass_advise.argtypes = [C.c_uint64, i8p, C.c_int32, C.c_int32, C.c_int32, C.c_int32, C.c_int32, C.c_int32,
                                C.c_int32, C.c_int32, C.c_void_p, C.c_double, C.c_uint64, i32p, f64p, f64p]
    lib.jass_advise.restype = C.c_int32
    lib.jass_agent_card.argtypes = [C.c_void_p, C.c_uint64, i8p, C.c_int32, C.c_int32, C.c_int32, C.c_int32,
                                     C.c_int32, C.c_int32, C.c_uint64]
    lib.jass_agent_card.restype = C.c_int32
    lib.jass_agent_trump.argtypes = [C.c_void_p, C.c_uint64, C.c_int32, C.c_int32, C.c_int32, C.c_uint64]
    lib.jass_agent_trump.restype = C.c_int32
    lib.jass_trump_policy.argtypes = [C.c_uint64, C.c_int32, f64p]
    lib.jass_trump_policy.restype = C.c_int32
    lib.jass_bonus.argtypes = [u64p, C.c_int32, C.c_int32, C.c_int32, i32p, i32p, u64p]
    lib.jass_dd_value.argtypes = [u64p, i32p, C.c_int32, C.c_int32, C.c_int32, C.c_int32, C.c_int32]
    lib.jass_dd_value.restype = C.c_int32
    lib.jass_card_features_size.restype = C.c_int32
    lib.jass_trump_features_size.restype = C.c_int32
    i32p_ = i32p
    lib.jass_log_card_samples.argtypes = [C.c_int64, i32p_, i8p, i8p, i8p, i8p, i8p, f32p, u64p, i8p, C.c_int32]
    lib.jass_log_trump_samples.argtypes = [C.c_int64, i32p_, i8p, i8p, i8p, i8p, i8p, f32p, i8p, C.c_int32]
    lib.jass_env_card_features.argtypes = [C.c_void_p, C.c_int32, f32p]
    assert lib.jass_abi_version() == 1
    return lib


lib = _load()


def legal(hand: int, trick: list[int], trump: int) -> int:
    t = np.array(trick + [0] * (4 - len(trick)), dtype=np.int32)
    return lib.jass_legal(hand, t, len(trick), trump)


def trick_winner_pos(trick: list[int], trump: int) -> int:
    return lib.jass_trick_winner_pos(np.array(trick, dtype=np.int32), trump)


# status codes of replay_check
REPLAY_STATUS = ["ok", "not a deck permutation", "illegal card", "wrong first seat",
                 "winner mismatch", "points mismatch"]


def replay_check(d: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Replays logged games through the engine. Returns (status, first_error_position)."""
    n = len(d["trump"])
    status = np.zeros(n, np.int8)
    pos = np.zeros(n, np.int8)
    lib.jass_replay_check(n, *(np.ascontiguousarray(d[k]) for k in
                               ("cards", "first", "win", "points", "trump", "dealer")), status, pos)
    return status, pos


class Agent:
    """A C++ agent by spec: random | heuristic | pimc[:N[:T]] | net[:...] (see engine/include/agents.hpp)."""

    def __init__(self, spec: str):
        self.spec = spec
        self._p = lib.jass_agent_create(spec.encode())
        if not self._p:
            raise ValueError(f"unknown agent spec {spec!r}")

    def __del__(self):
        if getattr(self, "_p", None):
            lib.jass_agent_destroy(self._p)
            self._p = None


class VecEnv:
    """N Schieber rounds stepped in lock-step; zero Python loops per game."""

    def __init__(self, n: int, seed: int = 0, rules: str = "swisslos"):
        self.n = n
        self._p = lib.jass_env_create(n, seed)
        lib.jass_env_set_rules(self._p, RULES[rules])
        self._agents: dict[str, Agent] = {}
        self.seat = np.zeros(n, np.int8)
        self.legal_bits = np.zeros(n, np.uint64)
        self.hands = np.zeros((n, 4), np.uint64)
        self.history = np.zeros((n, 36), np.int8)
        self.info = np.zeros((n, 8), np.int16)
        self.done = np.zeros(n, np.int8)
        self.final_score = np.zeros((n, 2), np.int16)
        lib.jass_env_reset(self._p)
        self.observe()

    def observe(self) -> None:
        lib.jass_env_observe(self._p, self.seat, self.legal_bits, self.hands.reshape(-1),
                             self.history.reshape(-1), self.info.reshape(-1))

    def legal_mask(self) -> np.ndarray:
        """Boolean [n, 43] mask of legal actions."""
        bits = self.legal_bits[:, None] >> np.arange(N_ACTIONS, dtype=np.uint64)
        return (bits & np.uint64(1)).astype(bool)

    def builtin_actions(self, agent: str, mask: np.ndarray | None = None) -> np.ndarray:
        """Actions chosen by a C++ agent (by spec) for the envs in mask (default: all)."""
        if agent not in self._agents:
            self._agents[agent] = Agent(agent)
        actions = np.zeros(self.n, np.int16)
        m = np.ones(self.n, np.int8) if mask is None else np.ascontiguousarray(mask, np.int8)
        lib.jass_env_agent_actions(self._p, self._agents[agent]._p, m, actions)
        return actions

    def card_features(self, i: int) -> np.ndarray:
        x = np.zeros(lib.jass_card_features_size(), np.float32)
        lib.jass_env_card_features(self._p, i, x)
        return x

    def step(self, actions: np.ndarray, auto_reset: bool = True) -> int:
        illegal = lib.jass_env_step(self._p, np.ascontiguousarray(actions, np.int16), int(auto_reset),
                                    self.done, self.final_score.reshape(-1))
        self.observe()
        return illegal

    def close(self) -> None:
        if self._p:
            lib.jass_env_destroy(self._p)
            self._p = None

    def __del__(self):
        self.close()
