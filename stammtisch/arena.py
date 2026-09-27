"""Arena: duplicate-deal matches between agents, Elo ratings, leaderboard history, gating.

  python -m stammtisch arena                          # default gauntlet
  python -m stammtisch arena --agents random,heuristic,net --deals 5000
  python -m stammtisch gate "pimc:n=48" --vs pimc --deals 400   # exit 1 unless clearly better

Every pair plays the same deals twice with swapped seats (duplicate), so card luck cancels.
Score = slate points per round (Swisslos rules: Weis, Stöck, multipliers), with 95 % CI.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import subprocess
import time
from datetime import datetime

from .engine import ROOT
from .monitor import SIM

OUT = ROOT / "runs" / "arena"
DOCS = ROOT / "docs" / "data"
DEFAULT_AGENTS = ["random", "heuristic", "net", "pimc:n=16,t=8", "pimc"]
ANCHOR = ("heuristic", 1500.0)


def is_slow(spec: str) -> bool:
    return spec.startswith("pimc")


def default_deals(a: str, b: str, fast: int = 20000, slow: int = 300) -> int:
    return slow if is_slow(a) or is_slow(b) else fast


def match(a: str, b: str, deals: int, threads: int | None = None, rules: str = "swisslos", seed: int = 1) -> dict:
    env = dict(os.environ, STAMMTISCH_MODELS=str(ROOT / "models"))
    cmd = [str(SIM), "--a", a, "--b", b, "--games", str(deals), "--rules", rules, "--seed", str(seed),
           "--interval", "30", "--threads", str(threads or os.cpu_count() or 1)]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {out.stderr.strip()}")
    s = json.loads(out.stdout.strip().splitlines()[-1])
    s["wall_s"] = time.time() - t0
    return s


def elo(results: list[dict]) -> dict[str, float]:
    """Bradley-Terry ratings from round wins (draws ignored), anchored at the heuristic."""
    agents = sorted({r["agent_a"] for r in results} | {r["agent_b"] for r in results})
    r = {a: 0.0 for a in agents}
    for _ in range(2000):  # minorization-maximization on strengths
        gamma = {a: math.exp(v) for a, v in r.items()}
        new = {}
        for a in agents:
            wins = sum(x["wins_a"] for x in results if x["agent_a"] == a) + \
                   sum(x["wins_b"] for x in results if x["agent_b"] == a)
            denom = 0.0
            for x in results:
                if a in (x["agent_a"], x["agent_b"]):
                    o = x["agent_b"] if x["agent_a"] == a else x["agent_a"]
                    denom += (x["wins_a"] + x["wins_b"]) / (gamma[a] + gamma[o])
            new[a] = math.log(max(wins, 0.5) / denom) if denom else 0.0
        mean = sum(new.values()) / len(new)
        r = {a: v - mean for a, v in new.items()}
    base = r.get(ANCHOR[0], 0.0)
    return {a: ANCHOR[1] + (v - base) * 400 / math.log(10) for a, v in r.items()}


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=ROOT).stdout.strip() or "none"
    except OSError:
        return "none"


def gauntlet(agents: list[str], deals: int | None = None, slow_deals: int | None = None) -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    results = []
    for a, b in itertools.combinations(agents, 2):
        n = default_deals(a, b, deals or 20000, slow_deals or 300)
        s = match(a, b, n)
        results.append(s)
        print(f"{a:>18} vs {b:<18} {s['diff_per_round']:+7.2f} ± {s['diff_ci95']:5.2f}  "
              f"round wins {s['wins_a']}:{s['wins_b']}  ({n} deals, {s['wall_s']:.0f}s)", flush=True)
    ratings = elo(results)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    rev = git_rev()
    (OUT / f"{datetime.now():%Y%m%d-%H%M%S}.json").write_text(
        json.dumps({"date": stamp, "git": rev, "results": results, "elo": ratings}, indent=1))
    # Semicolon-separated: agent specs contain commas.
    with open(DOCS / "leaderboard.csv", "w") as f:
        f.write("rank;agent;elo\n")
        for i, (a, v) in enumerate(sorted(ratings.items(), key=lambda x: -x[1]), 1):
            f.write(f"{i};{a};{v:.0f}\n")
    with open(DOCS / "matches.csv", "w") as f:
        f.write("a;b;deals;diff;ci95;wins_a;wins_b\n")
        for s in results:
            f.write(f"{s['agent_a']};{s['agent_b']};{s['deals']};{s['diff_per_round']:.1f};{s['diff_ci95']:.1f};"
                    f"{s['wins_a']};{s['wins_b']}\n")
    hist = DOCS / "leaderboard_history.csv"
    new = not hist.exists()
    with open(hist, "a") as f:
        if new:
            f.write("date;git;agent;elo\n")
        for a, v in ratings.items():
            f.write(f"{stamp};{rev};{a};{v:.0f}\n")
    print("\nElo (heuristic = 1500):")
    for a, v in sorted(ratings.items(), key=lambda x: -x[1]):
        print(f"  {v:7.0f}  {a}")
    return results


def gate(candidate: str, baseline: str, deals: int) -> bool:
    """True if the candidate beats the baseline with the 95 % CI above zero."""
    s = match(candidate, baseline, deals)
    lo = s["diff_per_round"] - s["diff_ci95"]
    ok = lo > 0
    print(f"{candidate} vs {baseline}: {s['diff_per_round']:+.2f} ± {s['diff_ci95']:.2f} per round "
          f"over {deals} duplicate deals -> {'PASS' if ok else 'FAIL'}")
    return ok
