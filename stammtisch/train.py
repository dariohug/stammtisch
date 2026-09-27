"""Imitation learning on the Swisslos logs: card policy and trump policy (P3).

Samples are generated on the fly by the C++ engine (same feature code the agents use).
Progress is appended to runs/train_<name>.jsonl; models go to models/<name>.{pt,bin}.

  python -m stammtisch train card  --steps 30000
  python -m stammtisch train trump --epochs 4
"""
from __future__ import annotations

import json
import queue
import struct
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import data
from .engine import ROOT, lib

MODELS = ROOT / "models"
RUNS = ROOT / "runs"


def mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def export(model: nn.Sequential, path: Path) -> None:
    """Binary format read by engine/src/net.cpp: 'JNN1', n_layers, then per layer
    in, out, W[out][in], b[out] (float32, little endian). ReLU between layers."""
    linears = [m for m in model if isinstance(m, nn.Linear)]
    with open(path, "wb") as f:
        f.write(b"JNN1")
        f.write(struct.pack("<i", len(linears)))
        for lin in linears:
            w = lin.weight.detach().cpu().numpy().astype(np.float32)
            b = lin.bias.detach().cpu().numpy().astype(np.float32)
            f.write(struct.pack("<ii", w.shape[1], w.shape[0]))
            f.write(w.tobytes())
            f.write(b.tobytes())


def split(d: dict[str, np.ndarray], val_frac: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Train/validation split by date: the latest games are held out."""
    order = np.argsort(d["date"], kind="stable")
    n_val = int(len(order) * val_frac)
    return np.sort(order[:-n_val]), np.sort(order[-n_val:])


def legal_mask(bits: np.ndarray) -> np.ndarray:
    return ((bits[:, None] >> np.arange(36, dtype=np.uint64)) & np.uint64(1)).astype(bool)


class CardBatches:
    """Background producer of (X, mask, y) batches of random (game, card) decisions."""

    def __init__(self, d, games: np.ndarray, batch: int, seed: int, threads: int = 4, prefetch: int = 6):
        self.d = {k: np.ascontiguousarray(d[k]) for k in ("cards", "dealer", "trump", "forehand")}
        self.games, self.batch, self.threads = games, batch, threads
        self.rng = np.random.default_rng(seed)
        self.q: queue.Queue = queue.Queue(prefetch)
        self.nf = lib.jass_card_features_size()
        threading.Thread(target=self._run, daemon=True).start()

    def make(self, g: np.ndarray, pos: np.ndarray):
        n = len(g)
        X = np.zeros((n, self.nf), np.float32)
        legal = np.zeros(n, np.uint64)
        y = np.zeros(n, np.int8)
        lib.jass_log_card_samples(n, g.astype(np.int32), pos.astype(np.int8), self.d["cards"], self.d["dealer"],
                                  self.d["trump"], self.d["forehand"], X, legal, y, self.threads)
        return torch.from_numpy(X), torch.from_numpy(legal_mask(legal)), torch.from_numpy(y.astype(np.int64))

    def _run(self):
        while True:
            g = self.rng.choice(self.games, self.batch)
            pos = self.rng.integers(0, 36, self.batch)
            self.q.put(self.make(g, pos))

    def __iter__(self):
        while True:
            yield self.q.get()


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, -1e9)


def log(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)


def train_card(steps: int = 30000, batch: int = 2048, lr: float = 1e-3, hidden: int = 512,
               name: str = "card_policy", torch_threads: int = 10) -> None:
    torch.set_num_threads(torch_threads)
    torch.manual_seed(0)
    d = data.load()
    tr, va = split(d)
    nf = lib.jass_card_features_size()
    model = mlp([nf, hidden, hidden, hidden, 36])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    batches = CardBatches(d, tr, batch, seed=1)
    val = CardBatches(d, va, 1, seed=2)  # only used for make()
    vrng = np.random.default_rng(3)
    vg, vp = vrng.choice(va, 50000), vrng.integers(0, 36, 50000)
    Xv, Mv, yv = val.make(vg, vp)
    forced_v = Mv.sum(1) == 1

    RUNS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    logf = RUNS / f"train_{name}.jsonl"
    logf.write_text("")
    t0, seen, run_loss, run_acc = time.time(), 0, 0.0, 0.0
    for step, (X, M, y) in enumerate(batches, 1):
        logits = masked_logits(model(X), M)
        loss = nn.functional.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        seen += len(y)
        run_loss = 0.98 * run_loss + 0.02 * loss.item() if step > 1 else loss.item()
        run_acc = 0.98 * run_acc + 0.02 * (logits.argmax(1) == y).float().mean().item() if step > 1 else 0.0
        if step % 500 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                lv = masked_logits(model(Xv), Mv)
                vloss = nn.functional.cross_entropy(lv, yv).item()
                hit = lv.argmax(1) == yv
                vacc = hit.float().mean().item()
                vacc_choice = hit[~forced_v].float().mean().item()
            model.train()
            log(logf, {"step": step, "samples": seen, "loss": run_loss, "acc": run_acc, "val_loss": vloss,
                       "val_acc": vacc, "val_acc_nonforced": vacc_choice, "lr": sched.get_last_lr()[0],
                       "samples_per_s": seen / (time.time() - t0)})
            torch.save(model.state_dict(), MODELS / f"{name}.pt")
            export(model, MODELS / f"{name}.bin")
        if step >= steps:
            break


def train_trump(epochs: int = 4, batch: int = 1024, lr: float = 1e-3, hidden: int = 256,
                name: str = "trump_policy", torch_threads: int = 4) -> None:
    torch.set_num_threads(torch_threads)
    torch.manual_seed(0)
    d = data.load()
    tr, va = split(d)
    arrs = {k: np.ascontiguousarray(d[k]) for k in ("cards", "dealer", "trump", "forehand")}
    nf = lib.jass_trump_features_size()

    def samples(games: np.ndarray):
        pushed = games[arrs["forehand"][games] == 0]
        g = np.concatenate([games, pushed]).astype(np.int32)
        which = np.concatenate([np.zeros(len(games), np.int8), np.ones(len(pushed), np.int8)])
        X = np.zeros((len(g), nf), np.float32)
        y = np.zeros(len(g), np.int8)
        lib.jass_log_trump_samples(len(g), g, which, arrs["cards"], arrs["dealer"], arrs["trump"],
                                   arrs["forehand"], X, y, 8)
        return torch.from_numpy(X), torch.from_numpy(y.astype(np.int64))

    Xt, yt = samples(tr)
    Xv, yv = samples(va)
    model = mlp([nf, hidden, hidden, 7])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(yt) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    logf = RUNS / f"train_{name}.jsonl"
    RUNS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    logf.write_text("")
    push_mask_v = Xv[:, 36] > 0.5  # after a push, pushing again is not allowed
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(len(yt))
        for i in range(0, len(perm) - batch + 1, batch):
            idx = perm[i:i + batch]
            logits = model(Xt[idx])
            logits = logits.masked_fill((Xt[idx, 36] > 0.5)[:, None] & (torch.arange(7) == 6), -1e9)
            loss = nn.functional.cross_entropy(logits, yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
        model.eval()
        with torch.no_grad():
            lv = model(Xv).masked_fill(push_mask_v[:, None] & (torch.arange(7) == 6), -1e9)
            rec = {"epoch": ep + 1, "step": step, "train_loss": loss.item(),
                   "val_loss": nn.functional.cross_entropy(lv, yv).item(),
                   "val_acc": (lv.argmax(1) == yv).float().mean().item(),
                   "val_acc_forehand": (lv.argmax(1) == yv)[~push_mask_v].float().mean().item(),
                   "val_acc_after_push": (lv.argmax(1) == yv)[push_mask_v].float().mean().item()}
        model.train()
        log(logf, rec)
        torch.save(model.state_dict(), MODELS / f"{name}.pt")
        export(model, MODELS / f"{name}.bin")


def load_selfplay(path: Path) -> dict[str, np.ndarray]:
    """jass-sim --log records -> arrays in the Swisslos layout (plain rules)."""
    raw = np.fromfile(path, dtype=np.int8).reshape(-1, 44)
    return {"cards": np.ascontiguousarray(raw[:, 4:40]), "dealer": np.ascontiguousarray(raw[:, 0]),
            "trump": np.ascontiguousarray(raw[:, 1]), "forehand": np.ascontiguousarray((raw[:, 2] == 0).astype(np.int8)),
            "team_a": raw[:, 3].copy()}


def distill(selfplay: str = "runs/selfplay.bin", steps: int = 20000, batch: int = 2048, lr: float = 3e-4,
            human_frac: float = 0.25, init: str = "card_policy", name: str = "card_policy_strong",
            torch_threads: int = 10) -> None:
    """Expert iteration: fine-tune the card policy on the search agent's own decisions
    (all seats of `jass-sim --a pimc --b pimc --log ...`), mixed with human data."""
    torch.set_num_threads(torch_threads)
    torch.manual_seed(1)
    sp = load_selfplay(ROOT / selfplay)
    n_games = len(sp["trump"])
    hold = max(1, n_games // 20)  # last 5 % for validation
    human = data.load()
    h_tr, _ = split(human)
    nf = lib.jass_card_features_size()
    hidden = 512
    model = mlp([nf, hidden, hidden, hidden, 36])
    model.load_state_dict(torch.load(MODELS / f"{init}.pt"))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    sp_batches = CardBatches(sp, np.arange(n_games - hold), int(batch * (1 - human_frac)), seed=5)
    h_batches = CardBatches(human, h_tr, batch - int(batch * (1 - human_frac)), seed=6)
    vg = np.random.default_rng(7).choice(np.arange(n_games - hold, n_games), 20000)
    Xv, Mv, yv = sp_batches.make(vg, np.random.default_rng(8).integers(0, 36, 20000))
    logf = RUNS / f"train_{name}.jsonl"
    logf.write_text("")
    t0 = time.time()
    for step, ((X1, M1, y1), (X2, M2, y2)) in enumerate(zip(sp_batches, h_batches), 1):
        X, M, y = torch.cat([X1, X2]), torch.cat([M1, M2]), torch.cat([y1, y2])
        loss = nn.functional.cross_entropy(masked_logits(model(X), M), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                lv = masked_logits(model(Xv), Mv)
                rec = {"step": step, "loss": loss.item(), "val_loss_search": nn.functional.cross_entropy(lv, yv).item(),
                       "val_acc_search": (lv.argmax(1) == yv).float().mean().item(),
                       "samples_per_s": step * batch / (time.time() - t0)}
            model.train()
            log(logf, rec)
            torch.save(model.state_dict(), MODELS / f"{name}.pt")
            export(model, MODELS / f"{name}.bin")
        if step >= steps:
            break
