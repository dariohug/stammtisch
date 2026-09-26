"""Numbers for the progress report: writes CSV files that docs/report.tex plots with pgfplots."""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from . import data, engine
from .engine import ROOT
from .monitor import SIM

OUT = ROOT / "docs" / "data"
MODES = ["Schellen", "Rosen", "Schilten", "Eichel", "Obenabe", "Undeufe"]


def _sim(*args: str) -> dict:
    out = subprocess.run([str(SIM), "--interval", "3600", *args], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def bench(seconds: float = 3.0) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cores = os.cpu_count() or 1
    threads = sorted({1, 2, 4, 6, 8, 10, 12, cores} & set(range(1, cores + 1)))
    rows = []
    with open(OUT / "scaling.csv", "w") as f:
        f.write("threads,heuristic,random\n")
        # Laptop throughput depends on turbo/power-limit state, so report best of 3 (capability).
        best = lambda agent, k: max((_sim("--seconds", str(seconds), "--threads", str(k), "--a", agent,  # noqa: E731
                                          "--b", agent) for _ in range(3)), key=lambda s: s["rounds_per_s"])
        for k in threads:
            h, r = best("heuristic", k), best("random", k)
            rows.append((k, h["rounds_per_s"], r["rounds_per_s"]))
            f.write(f"{k},{h['rounds_per_s']:.0f},{r['rounds_per_s']:.0f}\n")
            print(f"{k:>2} threads: heuristic {h['rounds_per_s']:>12,.0f}  random {r['rounds_per_s']:>12,.0f} rounds/s")
    results = {}
    with open(OUT / "agents.csv", "w") as f:
        f.write("a,b,deals,diff,ci95,winrate_a\n")
        for a, b in [("heuristic", "random"), ("random", "random"), ("heuristic", "heuristic")]:
            s = _sim("--games", "2000000", "--a", a, "--b", b)
            wr = s["wins_a"] / max(1, s["wins_a"] + s["wins_b"])
            results[a, b] = (s["diff_per_round"], s["diff_ci95"], wr)
            f.write(f"{a},{b},{s['deals']},{s['diff_per_round']:.3f},{s['diff_ci95']:.3f},{wr:.4f}\n")
            print(f"{a} vs {b}: {s['diff_per_round']:+.2f} ± {s['diff_ci95']:.2f} points/round, win rate {wr:.3f}")
    diff, ci, wr = results["heuristic", "random"]
    with open(OUT / "bench.tex", "w") as f:
        f.write(f"\\newcommand{{\\Cores}}{{{cores}}}\n")
        f.write(f"\\newcommand{{\\PeakRandom}}{{{max(r for _, _, r in rows) / 1e6:.1f}\\,M}}\n")
        f.write(f"\\newcommand{{\\PeakHeuristic}}{{{max(h for _, h, _ in rows) / 1e6:.1f}\\,M}}\n")
        f.write(f"\\newcommand{{\\SingleHeuristic}}{{{rows[0][1] / 1e3:.0f}\\,k}}\n")
        f.write(f"\\newcommand{{\\HeurDiff}}{{{diff:+.1f}\\,$\\pm$\\,{ci:.1f}}}\n")
        f.write(f"\\newcommand{{\\HeurWin}}{{{wr * 100:.1f}\\,\\%}}\n")


def _jass_kit_rejections(d: dict[str, np.ndarray]) -> int:
    """Human plays (4th card of a trick) that jass-kit would reject because of its trump-order bug."""
    strength = np.array([6, 5, 4, 8, 3, 7, 2, 1, 0])
    tricks = d["cards"].reshape(-1, 9, 4).astype(np.int16)
    trump = np.repeat(d["trump"], 9).reshape(-1, 9)
    suit, rank = tricks // 9, tricks % 9
    cand = (trump < 4) & (suit[..., 0] != trump) & (suit[..., 1] == trump) & (suit[..., 2] == trump)
    idx_best = np.maximum(tricks[..., 1], tricks[..., 2])
    rank_best = np.where(strength[rank[..., 1]] > strength[rank[..., 2]], tricks[..., 1], tricks[..., 2])
    cand &= idx_best != rank_best
    from jass.game.rule_schieber import RuleSchieber
    rule, rejected = RuleSchieber(), 0
    seats_of = lambda first: [(first - i) % 4 for i in range(4)]  # noqa: E731
    for g, k in zip(*np.nonzero(cand)):
        hand = np.zeros(36, np.int32)
        seat = seats_of(d["first"][g, k])[3]
        for kk in range(k, 9):  # cards this seat still holds at trick k
            hand[tricks[g, kk, seats_of(d["first"][g, kk]).index(seat)]] = 1
        valid = rule.get_valid_cards(hand, tricks[g, k, :3], 3, int(trump[g, k]))
        rejected += not valid[tricks[g, k, 3]]
    return rejected


def dataset_stats() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = data.load()
    n = len(d["trump"])
    status, _ = engine.replay_check(d)
    pts = data.team_points(d)
    decl = data.declarer_team(d)
    decl_pts = pts[np.arange(n), decl]
    pushed = d["forehand"] == 0

    with open(OUT / "trump.csv", "w") as f:
        f.write("mode,idx,share_direct,share_pushed,mean_decl_points\n")
        for m, name in enumerate(MODES):
            sel = d["trump"] == m
            f.write(f"{name},{m},{(sel & ~pushed).sum() / n * 100:.2f},{(sel & pushed).sum() / n * 100:.2f},"
                    f"{decl_pts[sel].mean():.2f}\n")

    edges = np.arange(0, 270, 10)
    hist, _ = np.histogram(decl_pts, bins=edges)
    with open(OUT / "decl_points.csv", "w") as f:
        f.write("bin,share\n")
        for lo, c in zip(edges[:-1], hist):
            f.write(f"{lo + 5},{c / n * 100:.3f}\n")

    rejected = _jass_kit_rejections(d)
    summary = {
        "games": int(n), "replay_ok": int((status == 0).sum()), "cards": int(n * 36),
        "players": int(len(np.unique(d["player_ids"][d["player_ids"] > 0]))),
        "push_rate": float(pushed.mean()), "decl_mean": float(decl_pts.mean()),
        "decl_win_rate": float((decl_pts > 157 / 2).mean()),
        "match_rate": float(((d["win"] % 2)[:, :, None] == np.arange(2)).all(1).any(1).mean()),
        "date_from": str(np.datetime64(int(d["date"].min()), "s").astype("datetime64[D]")),
        "date_to": str(np.datetime64(int(d["date"].max()), "s").astype("datetime64[D]")),
        "jasskit_rejected_human_plays": int(rejected),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    # LaTeX macros so the report never carries stale hand-typed numbers.
    with open(OUT / "numbers.tex", "w") as f:
        f.write(f"\\newcommand{{\\NGames}}{{{n:,}}}\n".replace(",", "'"))
        f.write(f"\\newcommand{{\\NCards}}{{{n * 36 / 1e6:.1f}\\,M}}\n")
        f.write(f"\\newcommand{{\\NReplayOK}}{{{summary['replay_ok']:,}}}\n".replace(",", "'"))
        f.write(f"\\newcommand{{\\NPlayers}}{{{summary['players']:,}}}\n".replace(",", "'"))
        f.write(f"\\newcommand{{\\PushRate}}{{{summary['push_rate'] * 100:.1f}\\,\\%}}\n")
        f.write(f"\\newcommand{{\\DeclMean}}{{{summary['decl_mean']:.1f}}}\n")
        f.write(f"\\newcommand{{\\DeclWin}}{{{summary['decl_win_rate'] * 100:.1f}\\,\\%}}\n")
        f.write(f"\\newcommand{{\\MatchRate}}{{{summary['match_rate'] * 100:.1f}\\,\\%}}\n")
        f.write(f"\\newcommand{{\\DateFrom}}{{{summary['date_from']}}}\n")
        f.write(f"\\newcommand{{\\DateTo}}{{{summary['date_to']}}}\n")
        f.write(f"\\newcommand{{\\JassKitRejected}}{{{rejected:,}}}\n".replace(",", "'"))
    print(json.dumps(summary, indent=2))
