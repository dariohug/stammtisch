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
TRUMP_NAMES = ["Schellen/♦", "Rosen/♥", "Schilten/♠", "Eichel/♣", "Obenabe", "Undeufe"]


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
    lib.jass_env_builtin_actions.argtypes = [C.c_void_p, C.c_int32, i8p, i16p]
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


class VecEnv:
    """N Schieber rounds stepped in lock-step; zero Python loops per game."""

    def __init__(self, n: int, seed: int = 0):
        self.n = n
        self._p = lib.jass_env_create(n, seed)
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
        actions = np.zeros(self.n, np.int16)
        m = np.ones(self.n, np.int8) if mask is None else mask.astype(np.int8)
        lib.jass_env_builtin_actions(self._p, {"random": 0, "heuristic": 1}[agent], m, actions)
        return actions

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
