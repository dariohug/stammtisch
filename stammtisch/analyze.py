"""Review a recorded game: for every decision of the chosen player, how much did the
played card cost compared with the best alternative?

Two views per decision:
  * with what you knew  - PIMC: the unseen cards are sampled consistently with everything
    visible at that moment (own hand, played cards, voids, shown Weis); each sample is played
    out by the imitation network (human-like play) and solved exactly for the last tricks.
  * with all cards open - the same evaluation on the true deal (hindsight).
Only the first view is fair to judge a decision; the second shows what was possible.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from . import engine
from .engine import ROOT, lib, mask_to_cards, trick_winner_pos
from .record import Game, pretty

TRUMP_LABEL = ["Eichel ♦", "Rosen ♥", "Schilten ♠", "Schellen ♣", "Obenabe", "Undeufe"]
MULT = [1, 1, 2, 2, 3, 3]

# Expected card points lost (from the player's own view) → verdict.
VERDICTS = [(1.0, "✓", "good"), (4.0, "?!", "inaccuracy"), (10.0, "?", "mistake"), (1e9, "??", "blunder")]


def verdict(loss: float) -> tuple[str, str]:
    for limit, sym, word in VERDICTS:
        if loss < limit:
            return sym, word
    return VERDICTS[-1][1:]


@dataclass
class Decision:
    index: int            # position in play order (0..35)
    trick: int            # 1..9
    player: int           # index into game.players
    played: int
    options: list[tuple[int, float, float]]  # (card, value with own info, value with open cards), best first
    loss: float           # expected card points lost vs. the best option (own info)
    loss_open: float      # same with all cards open (only meaningful when exact)
    forced: bool
    exact: bool           # open-cards value is an exact solution (late in the round)
    loss_se: float = 0.0  # standard error of `loss` (sampling noise)

    @property
    def clear(self) -> bool:
        """The loss is clearly larger than the sampling noise."""
        return self.loss > 2 * self.loss_se


@dataclass
class TrumpDecision:
    player: int
    pushed_before: bool   # decision after the partner pushed
    choice: int           # 0..5, or 6 = push
    values: list[float]   # expected slate difference per option (7; push nan when not allowed)
    values_open: list[float]  # with the true deal (modes only)
    loss: float
    human: list[float] | None = None  # how often Swisslos players choose each option with this hand


@dataclass
class Analysis:
    game: Game
    decisions: list[Decision]
    trump: list[TrumpDecision]
    score: tuple[int, int]         # final slate points of team of me / opponents
    samples: int
    rollout: str
    belief: float = 0.0


def _seats_of_cards(game: Game) -> list[int]:
    """Player index (into game.players) of every played card."""
    first = (game.dealer + 1) % 4
    out = []
    for t in range(0, len(game.cards), 4):
        trick = game.cards[t:t + 4]
        out += [(first + i) % 4 for i in range(len(trick))]
        if len(trick) == 4:
            first = (first + trick_winner_pos(trick, game.trump)) % 4
    return out


def _rollout_agent() -> engine.Agent | None:
    os.environ.setdefault("STAMMTISCH_MODELS", str(ROOT / "models"))
    try:
        return engine.Agent("net")
    except ValueError:
        return None


def analyze(game: Game, samples: int = 64, exact_left: int = 6, who: str = "me", seed: int = 1,
            belief: float = 0.5) -> Analysis:
    n = len(game.cards)
    if len(set(game.cards)) != n:
        raise ValueError("a card appears twice")
    owners = _seats_of_cards(game)
    for k, (lab, own) in enumerate(zip(game.labels, owners)):
        if lab is not None and lab != own:
            raise ValueError(f"trick {k // 4 + 1}: card {pretty(game.cards[k])} is labelled {game.players[lab]}, "
                             f"but by the rules {game.players[own]} plays it (leader = previous trick's winner)")
    hands_in = None
    if n < 36 or game.hands:
        if len(game.hands) != 4 or any(len(h) != 9 for h in game.hands.values()):
            raise ValueError("an incomplete game needs 'hands:' with 9 cards for each player")
        hands_in = np.zeros(4, np.uint64)
        for p, cards in game.hands.items():
            for c in cards:
                hands_in[Game.seat(p)] |= np.uint64(1) << np.uint64(c)

    team_me = {game.me, (game.me + 2) % 4}
    chosen = {"me": {game.me}, "team": team_me, "all": {0, 1, 2, 3}}[who]
    seat_mask = sum(1 << Game.seat(p) for p in chosen)
    agent = _rollout_agent()
    rollout_ptr = agent._p if agent else None
    rules = engine.RULES[game.rules]

    legal = np.zeros(36, np.uint64)
    info = np.zeros(36 * 36, np.float32)
    dd = np.zeros(36 * 36, np.float32)
    se = np.zeros(36 * 36, np.float32)
    cards = np.full(36, -1, np.int8)
    cards[:n] = game.cards
    res = lib.jass_analyze_round(cards, n, hands_in.ctypes.data if hands_in is not None else None,
                                 Game.seat(game.dealer), game.trump, int(game.pushed), rules, seat_mask, samples,
                                 exact_left, rollout_ptr, belief if agent else 0.0, seed, legal, info, dd, se)
    if res == -1:
        raise ValueError("cards and hands are inconsistent")
    if res > 0:
        k = res - 1
        raise ValueError(f"trick {k // 4 + 1}: {game.players[owners[k]]} may not play {pretty(game.cards[k])} "
                         "(must follow suit / no under-trumping)")
    info, dd, se = info.reshape(36, 36), dd.reshape(36, 36), se.reshape(36, 36)

    decisions = []
    for k in range(n):
        if not legal[k]:
            continue
        played = game.cards[k]
        # Best first; the played card wins ties so that "best" never contradicts a zero loss.
        opts = sorted(((c, float(info[k, c]), float(dd[k, c])) for c in mask_to_cards(int(legal[k]))),
                      key=lambda o: (-round(o[1], 3), o[0] != played))
        v_played = float(info[k, played])
        forced = len(opts) == 1
        loss = 0.0 if forced else max(0.0, (opts[0][1] - v_played) / 2)
        loss_open = 0.0 if forced else max(0.0, (max(o[2] for o in opts) - float(dd[k, played])) / 2)
        exact = 9 - k // 4 <= exact_left
        loss_se = 0.0 if forced or np.isnan(se[k, played]) else float(se[k, played]) / 2
        decisions.append(Decision(k, k // 4 + 1, owners[k], played, opts, loss, loss_open, forced, exact, loss_se))

    trump_decisions = _analyze_trump(game, chosen, samples, exact_left, rollout_ptr, rules, seed,
                                     belief if agent else 0.0)

    # Final score (slate points) from the engine's scoring.
    score = _final_score(game, rules)
    return Analysis(game, decisions, trump_decisions, score, samples, "net" if agent else "heuristic",
                    belief if agent else 0.0)


def _deal(game: Game) -> np.ndarray:
    hands = np.zeros(4, np.uint64)
    if game.hands:
        for p, cards in game.hands.items():
            for c in cards:
                hands[Game.seat(p)] |= np.uint64(1) << np.uint64(c)
        return hands
    for c, p in zip(game.cards, _seats_of_cards(game)):
        hands[Game.seat(p)] |= np.uint64(1) << np.uint64(c)
    return hands


def _analyze_trump(game, chosen, samples, exact_left, rollout_ptr, rules, seed, belief) -> list[TrumpDecision]:
    hands = _deal(game)
    forehand = (game.dealer + 1) % 4
    steps = [(forehand, False, 6 if game.pushed else game.trump)]
    if game.pushed:
        steps.append(((forehand + 2) % 4, True, game.trump))
    out = []
    for player, pushed_before, choice in steps:
        if player not in chosen:
            continue
        vi = np.zeros(7)
        vo = np.zeros(7)
        lib.jass_analyze_trump(int(hands[Game.seat(player)]), hands.ctypes.data, Game.seat(game.dealer),
                               int(pushed_before), rules, max(16, samples // 2), exact_left, rollout_ptr, belief,
                               seed + 7, vi, vo)
        vals = list(vi)
        if pushed_before:
            vals[6] = float("nan")
        best = max(v for v in vals if v == v)
        human = np.zeros(7)
        has_human = lib.jass_trump_policy(int(hands[Game.seat(player)]), int(pushed_before), human) == 0
        out.append(TrumpDecision(player, pushed_before, choice, vals, list(vo[:6]),
                                 max(0.0, (best - vals[choice]) / 2), list(human) if has_human else None))
    return out


def _final_score(game: Game, rules: int) -> tuple[int, int]:
    """Slate points (team of me, opponents) of a complete round, incl. Weis/Stöck/multiplier."""
    if len(game.cards) < 36:
        return (0, 0)
    owners = _seats_of_cards(game)
    mine = {game.me, (game.me + 2) % 4}
    pts, tricks = [0, 0], [0, 0]
    for t in range(9):
        trick = game.cards[4 * t:4 * t + 4]
        team = 0 if owners[4 * t + trick_winner_pos(trick, game.trump)] in mine else 1
        pts[team] += sum(_points(c, game.trump) for c in trick) + (5 if t == 8 else 0)
        tricks[team] += 1
    bonus = np.zeros(2, np.int32)
    weis_team = np.zeros(1, np.int32)
    shown = np.zeros(4, np.uint64)
    lib.jass_bonus(_deal(game), Game.seat(game.dealer), game.trump, rules, bonus, weis_team, shown)
    team_me = Game.seat(game.me) % 2  # engine team of me
    mult = MULT[game.trump] if rules == 0 else 1
    s_me = pts[0] + (100 if tricks[0] == 9 else 0) + int(bonus[team_me])
    s_opp = pts[1] + (100 if tricks[1] == 9 else 0) + int(bonus[1 - team_me])
    return s_me * mult, s_opp * mult


_POINTS_PLAIN = [11, 4, 3, 2, 10, 0, 0, 0, 0]
_POINTS_TRUMP = [11, 4, 3, 20, 10, 14, 0, 0, 0]
_POINTS_OBE = [11, 4, 3, 2, 10, 0, 8, 0, 0]
_POINTS_UNE = [0, 4, 3, 2, 10, 0, 8, 0, 11]


def _points(card: int, trump: int) -> int:
    r = card % 9
    if trump == 4:
        return _POINTS_OBE[r]
    if trump == 5:
        return _POINTS_UNE[r]
    return _POINTS_TRUMP[r] if card // 9 == trump else _POINTS_PLAIN[r]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _fmt_delta(v: float) -> str:
    return "±0" if abs(v) < 0.05 else f"{v:+.1f}"


def explain(a: Analysis, d: Decision, style: str = "german") -> str:
    """One-line reason for a mistake, from the actual trick (other cards as they were played)."""
    g = a.game
    t0 = (d.index // 4) * 4
    trick = list(g.cards[t0:t0 + 4])
    if len(trick) < 4:
        return ""
    pos = d.index - t0
    owners = _seats_of_cards(g)[t0:t0 + 4]
    mine = {d.player, (d.player + 2) % 4}
    best = d.options[0][0]
    alt = trick.copy()
    alt[pos] = best

    def outcome(cards):
        w = owners[trick_winner_pos(cards, g.trump)]
        return w in mine, sum(_points(c, g.trump) for c in cards) + (5 if t0 == 32 else 0)

    b, p = pretty(best, style), pretty(d.played, style)
    is_trump = lambda c: g.trump < 4 and c // 9 == g.trump  # noqa: E731
    if pos == 0:
        # A different lead changes everyone's reply, so the actual trick says nothing.
        gone = set(g.cards[:t0])
        stronger_left = [c for c in range(best // 9 * 9, best // 9 * 9 + 9)
                         if c not in gone and c != best and c not in _deal_cards(a, d.player)
                         and _beats(c, best, g.trump)]
        if is_trump(best) and not is_trump(d.played):
            return f"leading trump ({b}) was stronger than opening {p}'s suit"
        if not stronger_left:
            return f"{b} was a sure winner to lead (the highest card left in its suit)"
        return f"leading {b} instead of {p} gives your team better chances in this and the next tricks"
    won_now, pts_now = outcome(trick)
    won_alt, pts_alt = outcome(alt)
    as_played = "" if pos == 3 else " (if the others had played the same cards)"
    if won_alt and not won_now:
        return f"{b} would have taken this {pts_alt}-point trick for your team{as_played}"
    if won_now and won_alt and pts_alt > pts_now:
        return f"schmieren: {b} gives your team {pts_alt - pts_now} more points in this trick"
    if not won_now and not won_alt and pts_now > pts_alt:
        return f"{p} gave the opponents {pts_now - pts_alt} points more than {b}"
    if is_trump(d.played) and not is_trump(best):
        return f"trumping with {p} costs a trump you need later; {b} keeps it"
    if won_now and not won_alt:
        return f"taking this trick with {p} is worse later on; {b} lets it go and keeps the stronger card"
    if not is_trump(d.played) and is_trump(best):
        return f"{b} (trump) was the stronger play here"
    return f"{b} keeps better chances in the following tricks"


def _deal_cards(a: Analysis, player: int) -> set[int]:
    """Cards of `player` in the recorded deal."""
    owners = _seats_of_cards(a.game)
    if a.game.hands:
        return set(a.game.hands.get(player, []))
    return {c for c, o in zip(a.game.cards, owners) if o == player}


def _beats(c: int, other: int, trump: int) -> bool:
    """Does c beat `other` of the same suit under this trump mode?"""
    # strength order mirrors engine/src/dd.cpp
    trump_order = [3, 5, 0, 1, 2, 4, 6, 7, 8]
    if trump < 4 and c // 9 == trump:
        return trump_order.index(c % 9) < trump_order.index(other % 9)
    if trump == 5:
        return c % 9 > other % 9
    return c % 9 < other % 9


def render(a: Analysis, show_all_options: bool = False, style: str = "german") -> None:
    from rich.console import Console
    from rich.table import Table

    con = Console()
    g = a.game
    p = g.players
    partner = p[(g.me + 2) % 4]
    opp = f"{p[(g.me + 1) % 4]} & {p[(g.me + 3) % 4]}"
    declarer = (g.dealer + 1) % 4 if not g.pushed else (g.dealer + 3) % 4
    con.rule(f"[bold]Stammtisch review[/] {g.title}")
    con.print(f"[bold]{p[g.me]}[/] & {partner} vs {opp}   dealer {p[g.dealer]}   "
              f"trump [bold]{TRUMP_LABEL[g.trump]}[/] (x{MULT[g.trump] if g.rules == 'swisslos' else 1}) "
              f"by {p[declarer]}{' after push' if g.pushed else ''}   rules {g.rules}")
    if a.score != (0, 0):
        con.print(f"Final score (slate): [bold]{a.score[0]}[/] : {a.score[1]}")
    con.print(f"[dim]Evaluation: {a.samples} sampled deals per decision"
              + (f" (weighted by what the others' play revealed, belief {a.belief:g})" if a.belief else "")
              + f", rollouts by the {a.rollout} policy, "
              "exact play for the last tricks. Loss = expected card points your team gave away "
              "compared with the best card, judged only on what the player could see. "
              "'All cards open' (exact, last tricks only) shows what was possible in hindsight.[/]")

    for td in a.trump:
        t = Table(title=f"Trump decision of {p[td.player]}" + (" (after push)" if td.pushed_before else ""),
                  show_lines=False)
        t.add_column("option")
        t.add_column("expected slate diff (own view)", justify="right")
        t.add_column("with all cards open", justify="right")
        if td.human:
            t.add_column("players choose", justify="right")
        names = TRUMP_LABEL + ["push (schieben)"]
        best = max(v for v in td.values if v == v)
        for i, v in sorted(enumerate(td.values), key=lambda x: -x[1] if x[1] == x[1] else 1e9):
            if v != v:
                continue
            mark = " ← chosen" if i == td.choice else ""
            vo = f"{td.values_open[i]:+.0f}" if i < 6 else "–"
            style_ = "bold green" if v == best else ("yellow" if i == td.choice else "")
            row = [names[i] + mark, f"{v:+.1f}", vo] + ([f"{td.human[i] * 100:.0f} %"] if td.human else [])
            t.add_row(*row, style=style_)
        con.print(t)
        sym, word = verdict(td.loss)
        con.print(f"  → {sym} {word}: {td.loss:.1f} card points below the best option\n")

    t = Table(title="Card decisions", show_lines=False)
    for col, just in [("#", "right"), ("player", "left"), ("played", "left"), ("best (own view)", "left"),
                      ("loss", "right"), ("", "center"), ("all cards open: best / loss", "left")]:
        t.add_column(col, justify=just)
    total = 0.0
    for d in a.decisions:
        sym, word = verdict(d.loss)
        best = d.options[0]
        best_open = max(d.options, key=lambda o: o[2])
        colour = {"good": "", "inaccuracy": "yellow", "mistake": "dark_orange", "blunder": "bold red"}[word]
        if word != "good" and not d.clear:
            sym, colour = sym + "~", "dim"  # within sampling noise
        t.add_row(f"{d.trick}.{d.index % 4 + 1}", p[d.player], pretty(d.played, style),
                  "forced" if d.forced else pretty(best[0], style),
                  f"{d.loss:.1f}" + (f" ±{d.loss_se:.1f}" if d.loss >= 1.0 else ""), sym,
                  f"{pretty(best_open[0], style)} / {d.loss_open:.1f}" if d.exact and not d.forced else "",
                  style=colour)
        total += d.loss
    con.print(t)

    worst = sorted((d for d in a.decisions if d.loss >= 1.0 and d.clear), key=lambda d: -d.loss)
    if worst:
        con.print("[bold]Where it went wrong[/]")
        for d in worst[:5]:
            best_v = d.options[0][1]
            alts = ", ".join(f"{pretty(c, style)} {_fmt_delta((v - best_v) / 2)}"
                             for c, v, _ in d.options[: (None if show_all_options else 4)])
            con.print(f"  trick {d.trick}: {p[d.player]} played [bold]{pretty(d.played, style)}[/] "
                      f"({_fmt_delta(-d.loss)} ±{d.loss_se:.1f}); options: {alts}")
            con.print(f"    [italic]{explain(a, d, style)}[/]")
    n = len([d for d in a.decisions if not d.forced])
    con.print(f"\n[bold]Summary:[/] {n} real decisions, expected loss {total:.1f} card points"
              + (f" (+{sum(t.loss for t in a.trump):.1f} in the trump choice)" if a.trump else "")
              + f"; clear mistakes: {sum(1 for d in a.decisions if verdict(d.loss)[1] in ('mistake', 'blunder') and d.clear)} "
              "(~ = within sampling noise; more --samples makes it sharper).")


def to_markdown(a: Analysis, style: str = "german") -> str:
    g, p = a.game, a.game.players
    lines = [f"# Review: {g.title or 'recorded game'}", "",
             f"Trump **{TRUMP_LABEL[g.trump]}**, dealer {p[g.dealer]}"
             + (", pushed" if g.pushed else "") + (f", final {a.score[0]} : {a.score[1]}" if a.score != (0, 0) else ""),
             "", "| # | player | played | best (own view) | loss | verdict | best with open cards |",
             "|---|---|---|---|---:|---|---|"]
    for d in a.decisions:
        sym, word = verdict(d.loss)
        best_open = max(d.options, key=lambda o: o[2])
        if word != "good" and not d.clear:
            sym, word = sym + "~", word + " (noisy)"
        hindsight = f"{pretty(best_open[0], style)} ({d.loss_open:.1f})" if d.exact and not d.forced else ""
        lines.append(f"| {d.trick}.{d.index % 4 + 1} | {p[d.player]} | {pretty(d.played, style)} | "
                     f"{'forced' if d.forced else pretty(d.options[0][0], style)} | {d.loss:.1f} ±{d.loss_se:.1f} | "
                     f"{sym} {word} | {hindsight} |")
    worst = sorted((d for d in a.decisions if d.loss >= 1.0 and d.clear), key=lambda d: -d.loss)
    if worst:
        lines += ["", "## Where it went wrong", ""]
        lines += [f"- Trick {d.trick}, {p[d.player]} played **{pretty(d.played, style)}** "
                  f"(−{d.loss:.1f}): {explain(a, d, style)}" for d in worst[:5]]
    return "\n".join(lines) + "\n"


def to_html(a: Analysis, style: str = "german") -> str:
    """Self-contained HTML review: every trick as a row of cards, decisions coloured by verdict."""
    import html as H
    g, p = a.game, a.game.players
    owners = _seats_of_cards(g)
    by_index = {d.index: d for d in a.decisions}
    suit_cls = ["ei", "ro", "schi", "sche"]
    colour = {"good": "good", "inaccuracy": "inacc", "mistake": "mist", "blunder": "blund"}

    def card(c: int, extra: str = "") -> str:
        return f'<span class="card {suit_cls[c // 9]} {extra}">{H.escape(pretty(c, style))}</span>'

    rows = []
    for t in range(0, len(g.cards), 4):
        cells = []
        trick = g.cards[t:t + 4]
        winner = owners[t + trick_winner_pos(trick, g.trump)] if len(trick) == 4 else None
        for i, c in enumerate(trick):
            k = t + i
            d = by_index.get(k)
            cls, note = "", ""
            if d is not None and not d.forced:
                word = verdict(d.loss)[1]
                cls = colour[word] if (word == "good" or d.clear) else "noisy"
                if word != "good":
                    alts = " ".join(card(o[0]) + f'<small>{(o[1] - d.options[0][1]) / 2:+.1f}</small>'
                                    for o in d.options[:3])
                    note = (f'<div class="note"><b>−{d.loss:.1f}</b> ±{d.loss_se:.1f} · {alts}'
                            f'<div class="why">{H.escape(explain(a, d, style))}</div></div>')
            who = H.escape(p[owners[k]]) + (" ★" if owners[k] == winner else "")
            cells.append(f'<div class="play {cls}"><div class="who">{who}</div>{card(c)}{note}</div>')
        rows.append(f'<div class="trick"><div class="tno">{t // 4 + 1}</div>{"".join(cells)}</div>')

    trump_html = ""
    for td in a.trump:
        names = TRUMP_LABEL + ["schieben"]
        items = sorted((i for i in range(7) if td.values[i] == td.values[i]), key=lambda i: -td.values[i])
        lis = "".join(f'<li class="{"chosen" if i == td.choice else ""}">{names[i]} <b>{td.values[i]:+.0f}</b>'
                      + (f' <small>({td.human[i] * 100:.0f}% of players)</small>' if td.human else "") + "</li>"
                      for i in items)
        trump_html += (f'<section><h2>Trump call of {H.escape(p[td.player])}'
                       f'{" after push" if td.pushed_before else ""}</h2><ol class="trump">{lis}</ol>'
                       f'<p>Expected loss of the call: <b>{td.loss:.1f}</b> card points.</p></section>')
    total = sum(d.loss for d in a.decisions)
    clear = sum(1 for d in a.decisions if verdict(d.loss)[1] in ("mistake", "blunder") and d.clear)
    score = f"{a.score[0]} : {a.score[1]}" if a.score != (0, 0) else "–"
    return f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Jass review</title><style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6b66;--line:#e4e1d9;--card:#fff;--good:#2e7d32;--inacc:#b58900;
--mist:#d9730d;--blund:#c62828;--ei:#8a5a00;--ro:#c62828;--schi:#1f6f3f;--sche:#b8860b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161614;--fg:#eceae4;--muted:#a09e96;--line:#34332f;--card:#22211e}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;max-width:960px;margin:0 auto;padding:16px}}
h1{{font-size:22px;margin:4px 0}} h2{{font-size:16px;margin:18px 0 6px}} .meta{{color:var(--muted)}}
.trick{{display:grid;grid-template-columns:28px repeat(4,minmax(0,1fr));gap:8px;padding:8px 0;border-top:1px solid var(--line)}}
.tno{{color:var(--muted);font-weight:600;padding-top:18px}} .play{{min-width:0;border-radius:8px;padding:4px 6px}}
.who{{font-size:12px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.card{{display:inline-block;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:2px 7px;
font-weight:600;margin:1px 2px 1px 0}} .ei{{color:var(--ei)}} .ro{{color:var(--ro)}} .schi{{color:var(--schi)}} .sche{{color:var(--sche)}}
.good{{box-shadow:inset 3px 0 0 var(--good)}} .inacc{{box-shadow:inset 3px 0 0 var(--inacc)}}
.mist{{box-shadow:inset 3px 0 0 var(--mist)}} .blund{{box-shadow:inset 3px 0 0 var(--blund)}} .noisy{{box-shadow:inset 3px 0 0 var(--line)}}
.note{{font-size:12px;margin-top:4px}} .note small{{color:var(--muted);margin-right:4px}} .why{{color:var(--muted);font-style:italic}}
ol.trump li.chosen{{font-weight:700}} .summary{{padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card)}}
@media (max-width:600px){{.trick{{grid-template-columns:22px repeat(2,minmax(0,1fr))}} .tno{{grid-row:span 2}}}}
</style></head><body>
<h1>{H.escape(g.title or "Jass review")}</h1>
<div class="meta">Trump <b>{TRUMP_LABEL[g.trump]}</b> · dealer {H.escape(p[g.dealer])}{" · pushed" if g.pushed else ""}
 · final {score} · {a.samples} samples per decision</div>
<p class="summary">Expected loss of the reviewed decisions: <b>{total:.1f}</b> card points; clear mistakes: <b>{clear}</b>.
Colours: green ✓, yellow inaccuracy, orange mistake, red blunder, grey within sampling noise. ★ = trick winner.</p>
{trump_html}<section><h2>Tricks</h2>{"".join(rows)}</section></body></html>
"""
