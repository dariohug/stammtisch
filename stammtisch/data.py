"""Swisslos game logs -> compact numpy arrays (data/processed/swisslos.npz)."""
from __future__ import annotations

import gzip
import json
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .engine import ROOT, card_index

RAW = ROOT / "data" / "raw" / "swisslos"
PROCESSED = ROOT / "data" / "processed" / "swisslos.npz"


def _parse_file(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            rec = json.loads(line)
            g = rec["game"]
            tricks = g["tricks"]
            if len(tricks) != 9 or g.get("jassTyp") != "SCHIEBER":
                continue
            rows.append((
                [card_index(c) for t in tricks for c in t["cards"]],
                [t["first"] for t in tricks],
                [t["win"] for t in tricks],
                [t["points"] for t in tricks],
                g["trump"], g["dealer"], g["forehand"],
                rec.get("player_ids", [0, 0, 0, 0]),
                int(datetime.strptime(rec["date"], "%d.%m.%y %H:%M:%S").timestamp()),
            ))
    cols = list(zip(*rows))
    return {
        "cards": np.array(cols[0], np.int8), "first": np.array(cols[1], np.int8),
        "win": np.array(cols[2], np.int8), "points": np.array(cols[3], np.int16),
        "trump": np.array(cols[4], np.int8), "dealer": np.array(cols[5], np.int8),
        "forehand": np.array(cols[6], np.int8), "player_ids": np.array(cols[7], np.int32),
        "date": np.array(cols[8], np.int64),
    }


def ingest(raw: Path = RAW, out: Path = PROCESSED, workers: int | None = None) -> dict[str, np.ndarray]:
    files = sorted(raw.glob("jass_game_*.txt.gz"))
    if not files:
        raise FileNotFoundError(f"no jass_game_*.txt.gz in {raw} - see data/README.md")
    with Pool(workers) as pool:
        parts = pool.map(_parse_file, files)
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **d)
    return d


def load(path: Path = PROCESSED) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def declarer_team(d: dict[str, np.ndarray]) -> np.ndarray:
    """Team (0/1) that declared trump: forehand seat, or its partner after a push."""
    forehand = (d["dealer"].astype(np.int16) + 3) % 4
    seat = np.where(d["forehand"] == 1, forehand, (forehand + 2) % 4)
    return seat % 2


def team_points(d: dict[str, np.ndarray]) -> np.ndarray:
    """[n, 2] round points per team incl. match bonus."""
    team = d["win"] % 2
    pts = np.stack([(d["points"] * (team == t)).sum(1) for t in (0, 1)], 1)
    tricks = np.stack([(team == t).sum(1) for t in (0, 1)], 1)
    return pts + 100 * (tricks == 9)
