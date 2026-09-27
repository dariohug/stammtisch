"""Live advice at the table: what to play (or which trump to call) from your own view.

  python -m stammtisch advise --hand "RoA RoK Ro9 SchiU Schi10 Ei7 Ei8 Sche6 ScheK" --dealer left
  python -m stammtisch advise --hand "..." --dealer left --trump Rosen --played "RoA Ro6 Ro7 Ro9  Ei6"

Seats are relative to you, in playing order: you, right (plays after you), partner, left.
--played lists every card of the round so far in play order, starting with the forehand.
"""
from __future__ import annotations

import re

import numpy as np
from rich.console import Console
from rich.table import Table

from . import engine
from .analyze import MULT, TRUMP_LABEL, _rollout_agent
from .engine import lib
from .record import TRUMP_WORDS, parse_card, pretty

SEAT_WORDS = {"me": 0, "ich": 0, "right": 3, "rechts": 3, "partner": 2, "left": 1, "links": 1}


def _cards(text: str) -> list[int]:
    return [parse_card(t) for t in re.split(r"[\s,|]+", text.strip()) if t]


def advise(hand: str, dealer: str, trump: str | None = None, pushed: bool = False, played: str = "",
           rules: str = "swisslos", samples: int = 64, exact_left: int = 6, belief: float = 0.5,
           seed: int = 1) -> None:
    con = Console()
    my = _cards(hand)
    seat_dealer = SEAT_WORDS[dealer.lower()]
    agent = _rollout_agent()
    ptr = agent._p if agent else None
    bel = belief if agent else 0.0
    rule_id = engine.RULES[rules]
    mask = sum(1 << c for c in my)

    if trump is None:
        forehand = (seat_dealer + 3) % 4
        chooser = (forehand + 2) % 4 if pushed else forehand
        if chooser != 0:
            raise SystemExit("it is not your trump call (check --dealer / --pushed)")
        if len(my) != 9:
            raise SystemExit("give all 9 cards of your hand")
        vi, vo = np.zeros(7), np.zeros(7)
        lib.jass_analyze_trump(mask, None, seat_dealer, int(pushed), rule_id, samples, exact_left, ptr, bel, seed,
                               vi, vo)
        mult = MULT if rules == "swisslos" else [1] * 6
        human = np.zeros(7)
        has_human = lib.jass_trump_policy(mask, int(pushed), human) == 0
        opts = [(TRUMP_LABEL[m] + (f" (x{mult[m]})" if rules == "swisslos" else ""), vi[m], vi[m] / mult[m], human[m])
                for m in range(6)]
        if not pushed:
            opts.append(("push (schieben)", vi[6], float("nan"), human[6]))
        opts.sort(key=lambda o: -o[1])
        t = Table(title="Trump call: expected point difference for your team")
        t.add_column("option")
        t.add_column("slate (with multiplier)", justify="right")
        t.add_column("vs best", justify="right")
        t.add_column("raw points", justify="right")
        if has_human:
            t.add_column("Swisslos players choose", justify="right")
        for i, (name, v, raw, h) in enumerate(opts):
            row = [name, f"{v:+.1f}", "best" if i == 0 else f"{v - opts[0][1]:+.1f}", "" if raw != raw else f"{raw:+.1f}"]
            if has_human:
                row.append(f"{h * 100:.0f} %")
            t.add_row(*row, style="bold green" if i == 0 else "")
        con.print(t)
        return

    tr = TRUMP_WORDS[trump.lower()]
    seq = _cards(played)
    arr = np.full(36, -1, np.int8)
    arr[:len(seq)] = seq
    cards = np.zeros(9, np.int32)
    values = np.zeros(9)
    se = np.zeros(9)
    # Your current hand: the cards you still hold (drop the ones you already played).
    mine_now = [c for c in my if c not in seq]
    n = lib.jass_advise(sum(1 << c for c in mine_now), arr, len(seq), 0, seat_dealer, tr, int(pushed), rule_id,
                        samples, exact_left, ptr, bel, seed, cards, values, se)
    if n == -2:
        raise SystemExit("it is not your turn after these cards (check --dealer and --played)")
    if n < 0:
        raise SystemExit("inconsistent input: your hand + your played cards must be 9 distinct cards")
    order = sorted(range(n), key=lambda i: -values[i])
    t = Table(title=f"Your move ({TRUMP_LABEL[tr]}, trick {len(seq) // 4 + 1})")
    t.add_column("card")
    t.add_column("vs best (card points)", justify="right")
    for rank, i in enumerate(order):
        d = (values[i] - values[order[0]]) / 2
        t.add_row(pretty(int(cards[i])), "best" if rank == 0 else f"{d:+.1f} ±{se[i] / 2:.1f}",
                  style="bold green" if rank == 0 else ("dim" if -d < se[i] else ""))
    con.print(t)
