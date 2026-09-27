"""Play a round against the engine in the terminal, with a coach.

  python -m stammtisch play                      # you + 3 search bots, coach on
  python -m stammtisch play --bots net --no-coach

You sit at seat 0; the others are, in playing order, Rechts (right-hand opponent), Partner, Links.
After each of your decisions the coach shows how the options compared (from your view).
Every round is saved to runs/games/<time>.jass, ready for `stammtisch analyze`.
"""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
from rich.console import Console
from rich.prompt import Prompt

from . import engine
from .analyze import MULT, TRUMP_LABEL, _rollout_agent
from .engine import ROOT, lib, mask_to_cards
from .record import TRUMP_WORDS, Game, dump, parse_card, pretty

NAMES = ["Du", "Links", "Partner", "Rechts"]  # by engine seat (next_seat = seat - 1: 0 -> 3 -> 2 -> 1)


def _hand_str(cards: list[int]) -> str:
    return "  ".join(pretty(c) for c in sorted(cards, key=lambda c: (c // 9, c % 9)))


class Coach:
    def __init__(self, rules: str, samples: int):
        self.agent = _rollout_agent()
        self.ptr = self.agent._p if self.agent else None
        self.rules, self.samples = rules, samples
        self.losses: list[float] = []

    def trump(self, hand: int, dealer: int, pushed: bool) -> np.ndarray:
        vi, vo = np.zeros(7), np.zeros(7)
        lib.jass_analyze_trump(hand, None, dealer, int(pushed), engine.RULES[self.rules], self.samples, 6,
                               self.ptr, 0.5 if self.ptr else 0.0, 11, vi, vo)
        return vi

    def cards(self, hand: int, played: list[int], dealer: int, trump: int, pushed: bool) -> dict[int, tuple]:
        """card -> (value, standard error vs. the best card)."""
        arr = np.full(36, -1, np.int8)
        arr[:len(played)] = played
        cards, values, se = np.zeros(9, np.int32), np.zeros(9), np.zeros(9)
        n = lib.jass_advise(hand, arr, len(played), 0, dealer, trump, int(pushed), engine.RULES[self.rules],
                            self.samples, 6, self.ptr, 0.5 if self.ptr else 0.0, 12, cards, values, se)
        return {int(cards[i]): (float(values[i]), float(se[i])) for i in range(max(n, 0))}


def play(bots: str = "pimc:n=24", rules: str = "swisslos", coach: bool = True, seed: int | None = None,
         samples: int = 48) -> None:
    os.environ.setdefault("STAMMTISCH_MODELS", str(ROOT / "models"))
    con = Console()
    seed = seed if seed is not None else int.from_bytes(os.urandom(4), "little")
    env = engine.VecEnv(1, seed=seed, rules=rules)
    ch = Coach(rules, samples) if coach else None
    dealer = int(env.info[0, 3])
    con.rule(f"[bold]Stammtisch[/] — Geber: {NAMES[dealer]}, Regeln: {rules}")
    my_hand0 = mask_to_cards(int(env.hands[0, 0]))
    con.print(f"Deine Karten: [bold]{_hand_str(my_hand0)}[/]")
    trump_calls: list[str] = []
    while True:
        seat = int(env.seat[0])
        trump, pushed = int(env.info[0, 0]), bool(env.info[0, 2])
        n_played = int(env.info[0, 4])
        if trump >= 0 and n_played % 4 == 0:
            con.print(f"[dim]— Stich {n_played // 4 + 1} —[/]")
        if seat != 0:
            action = int(env.builtin_actions(bots)[0])
            if trump < 0:
                trump_calls.append(f"{NAMES[seat]}: {'schiebt' if action == 42 else TRUMP_LABEL[action - 36]}")
                con.print(trump_calls[-1])
            else:
                con.print(f"  {NAMES[seat]:>8} spielt [bold]{pretty(action)}[/]")
            env.step(np.array([action], np.int16), auto_reset=False)
        elif trump < 0:
            options = ["eichel", "rosen", "schilten", "schellen", "obenabe", "undeufe"] + ([] if pushed else ["schieben"])
            choice = Prompt.ask("Trumpf?", choices=options, show_choices=True).lower()
            action = 42 if choice == "schieben" else 36 + TRUMP_WORDS[choice]
            if ch:
                v = ch.trump(int(env.hands[0, 0]), dealer, pushed)
                idx = 6 if action == 42 else action - 36
                allowed = [m for m in range(7) if not (pushed and m == 6)]
                best = max(allowed, key=lambda m: v[m])
                names = TRUMP_LABEL + ["schieben"]
                loss = (v[best] - v[idx]) / 2
                ch.losses.append(max(0.0, loss))
                verdict = "[green]✓ beste Wahl[/]" if idx == best or loss < 1 else \
                    f"[yellow]Coach: {names[best]} wäre besser gewesen (~{loss:.0f} Punkte)[/]"
                con.print(f"  {verdict}   [dim]" + ", ".join(f"{names[m]} {v[m]:+.0f}" for m in
                                                            sorted(allowed, key=lambda m: -v[m])[:4]) + "[/]")
            env.step(np.array([action], np.int16), auto_reset=False)
        else:
            if n_played < 4:
                con.print(f"Trumpf [bold]{TRUMP_LABEL[trump]}[/]" + (f" x{MULT[trump]}" if rules == "swisslos" else ""))
            hand = mask_to_cards(int(env.hands[0, 0]))
            legal = [c for c in range(36) if env.legal_mask()[0, c]]
            if len(legal) == 1:
                con.print(f"  {NAMES[0]:>8} spielt [bold]{pretty(legal[0])}[/] (einzige erlaubte Karte)")
                env.step(np.array([legal[0]], np.int16), auto_reset=False)
                if int(env.info[0, 4]) == 35:
                    last_seat = int(env.seat[0])
                    last = mask_to_cards(int(env.hands[0, last_seat]))[0]
                    history = [int(c) for c in env.history[0, :35]] + [last]
                    con.print(f"  {NAMES[last_seat]:>8} spielt [bold]{pretty(last)}[/]")
                    env.step(np.array([last], np.int16), auto_reset=False)
                    break
                continue
            con.print(f"Deine Karten: {_hand_str(hand)}   [dim](erlaubt: {' '.join(pretty(c) for c in legal)})[/]")
            advice = ch.cards(int(env.hands[0, 0]), [int(c) for c in env.history[0, :n_played]], dealer, trump,
                              pushed) if ch and len(legal) > 1 else {}
            while True:
                try:
                    card = parse_card(Prompt.ask("Deine Karte"))
                except ValueError as e:
                    con.print(f"[red]{e}[/]")
                    continue
                if card in legal:
                    break
                con.print("[red]nicht erlaubt (Farbe angeben / nicht untertrumpfen)[/]")
            con.print(f"  {NAMES[0]:>8} spielt [bold]{pretty(card)}[/]")
            if advice:
                best = max(advice, key=lambda c: advice[c][0])
                loss = (advice[best][0] - advice[card][0]) / 2
                noise = advice[card][1] / 2
                ch.losses.append(max(0.0, loss))
                if loss < 1 or loss <= 2 * noise:
                    con.print("  [green]✓[/]" + (f" [dim](Alternative {pretty(best)} kaum besser)[/]" if loss >= 1 else ""))
                else:
                    con.print(f"  [yellow]Coach: {pretty(best)} war ~{loss:.1f} ±{noise:.1f} Punkte besser[/]")
            env.step(np.array([card], np.int16), auto_reset=False)
        if int(env.info[0, 4]) == 35:
            # The last card is forced; play it for whoever holds it.
            last_seat = int(env.seat[0])
            last = mask_to_cards(int(env.hands[0, last_seat]))[0]
            history = [int(c) for c in env.history[0, :35]] + [last]
            con.print(f"  {NAMES[last_seat]:>8} spielt [bold]{pretty(last)}[/]")
            env.step(np.array([last], np.int16), auto_reset=False)
            break

    score = env.final_score[0]
    con.rule("Resultat")
    con.print(f"Dein Team: [bold]{int(score[0])}[/]   Gegner: {int(score[1])}")
    if ch and ch.losses:
        con.print(f"Coach: {sum(ch.losses):.1f} Punkte erwarteter Verlust über {len(ch.losses)} Entscheidungen")
    # Save for later review. Playing order starting with me: seat 0, 3, 2, 1.
    order = [0, 3, 2, 1]
    game = Game(players=[NAMES[s] for s in order], dealer=order.index(dealer), trump=int(env.info[0, 0]),
                pushed=bool(env.info[0, 2]), cards=history, me=0, rules=rules,
                title=f"gespielt am {datetime.now():%d.%m.%Y %H:%M}")
    out = ROOT / "runs" / "games"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{datetime.now():%Y%m%d-%H%M%S}.jass"
    path.write_text(dump(game), encoding="utf-8")
    con.print(f"gespeichert: {path.relative_to(ROOT)}  →  python -m stammtisch analyze {path.relative_to(ROOT)}")
