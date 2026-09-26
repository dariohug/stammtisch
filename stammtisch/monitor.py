"""Live terminal dashboard for jass-sim runs (also logs every heartbeat to runs/*.jsonl)."""
from __future__ import annotations

import json
import os
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path

import psutil
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from .engine import ROOT, TRUMP_NAMES

SIM = ROOT / "build" / "jass-sim"
RUNS = ROOT / "runs"
SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values, width: int = 48) -> str:
    vals = list(values)[-width:]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return "".join(SPARK[int((v - lo) / span * (len(SPARK) - 1))] for v in vals)


def bar(frac: float, width: int = 20) -> str:
    n = max(0, min(width, round(frac * width)))
    return "█" * n + "·" * (width - n)


def human(n: float) -> str:
    for unit in ("", "k", "M", "G"):
        if abs(n) < 1000:
            return f"{n:,.1f}{unit}" if unit else f"{n:,.0f}"
        n /= 1000
    return f"{n:.1f}T"


def cpu_temperature() -> float | None:
    try:
        temps = psutil.sensors_temperatures()
    except (AttributeError, OSError):
        return None
    for key in ("coretemp", "k10temp", "thinkpad", "acpitz"):
        if temps.get(key):
            return max(t.current for t in temps[key])
    return None


class Dashboard:
    def __init__(self, cmd: list[str]):
        self.cmd = cmd
        self.hb: dict = {}
        self.rates: deque[float] = deque(maxlen=240)
        self.proc_ps: psutil.Process | None = None
        self._system: Panel | None = None
        self._rss = 0.0

    def header(self) -> Panel:
        h = self.hb
        t = Text()
        t.append("stammtisch · jass-sim  ", style="bold")
        t.append(f"A={h.get('agent_a', '?')}  vs  B={h.get('agent_b', '?')}   ", style="cyan")
        t.append(f"threads={h.get('threads', '?')}  duplicate={h.get('duplicate', '?')}   ")
        t.append(f"elapsed {h.get('t', 0):.1f}s", style="dim")
        target = h.get("target") or 0
        secs = h.get("seconds") or 0
        frac = (h.get("deals", 0) / target) if target else (h.get("t", 0) / secs if secs else 0)
        frac = min(1.0, frac)
        return Panel(Group(t, ProgressBar(total=1.0, completed=frac, width=None)),
                     title=f"{frac * 100:5.1f}%", border_style="blue")

    def throughput(self) -> Panel:
        h = self.hb
        rate = h.get("rounds_per_s", 0)
        avg = h.get("rounds", 0) / max(h.get("t", 1e-9), 1e-9)
        tab = Table.grid(padding=(0, 2))
        tab.add_row("rounds/s (now)", Text(human(rate), style="bold green"))
        tab.add_row("rounds/s (avg)", human(avg))
        tab.add_row("cards/s  (avg)", human(avg * 36))
        tab.add_row("rounds total", human(h.get("rounds", 0)))
        return Panel(Group(tab, Text(sparkline(self.rates), style="green")), title="Throughput")

    def results(self) -> Panel:
        h = self.hb
        rounds = max(1, h.get("rounds", 0))
        tab = Table.grid(padding=(0, 2))
        tab.add_row("mean points A / B", f"{h.get('mean_points_a', 0):.2f} / {h.get('mean_points_b', 0):.2f}")
        tab.add_row("A − B per round", Text(f"{h.get('diff_per_round', 0):+.2f} ± {h.get('diff_ci95', 0):.2f}",
                                             style="bold"))
        tab.add_row("round win rate A", f"{h.get('wins_a', 0) / rounds * 100:.1f} %")
        tab.add_row("matches A / B", f"{h.get('matches_a', 0):,} / {h.get('matches_b', 0):,}")
        tab.add_row("push rate", f"{h.get('pushes', 0) / rounds * 100:.1f} %")
        trump = h.get("trump", [0] * 6)
        tot = max(1, sum(trump))
        for name, c in zip(TRUMP_NAMES, trump):
            tab.add_row(f"  {name}", f"{bar(c / tot, 16)} {c / tot * 100:4.1f}%")
        return Panel(tab, title="Results (95% CI, duplicate deals)")

    def system(self) -> Panel:
        # Once the simulator has exited, keep showing the last sample taken under load.
        if self._system is not None and self.hb.get("type") == "summary":
            return self._system
        tab = Table.grid(padding=(0, 1))
        per_cpu = psutil.cpu_percent(percpu=True)
        for i in range(0, len(per_cpu), 2):
            cells = [f"cpu{j:<2} {bar(per_cpu[j] / 100, 10)} {per_cpu[j]:5.1f}%" for j in (i, i + 1)
                     if j < len(per_cpu)]
            tab.add_row(*cells)
        freq = psutil.cpu_freq()
        temp = cpu_temperature()
        mem = psutil.virtual_memory()
        if self.proc_ps:
            try:
                self._rss = self.proc_ps.memory_info().rss / 2**20
            except psutil.Error:  # exited already; keep the last reading
                pass
        info = Text(f"freq {freq.current / 1000:.2f} GHz  " if freq else "")
        if temp is not None:
            info.append(f"temp {temp:.0f}°C  ", style="red" if temp > 90 else "yellow" if temp > 80 else "")
        info.append(f"RAM {mem.percent:.0f}%  sim RSS {self._rss:.1f} MiB")
        self._system = Panel(Group(tab, info), title="System")
        return self._system

    def threads(self) -> Panel:
        per = self.hb.get("per_thread_rounds", [])
        top = max(per) if per else 1
        tab = Table.grid(padding=(0, 1))
        for i, n in enumerate(per):
            tab.add_row(f"t{i:<2}", bar(n / top if top else 0, 24), human(n))
        return Panel(tab, title="Worker load")

    def render(self) -> Layout:
        lay = Layout()
        lay.split_column(Layout(self.header(), size=4), Layout(name="body"))
        lay["body"].split_row(Layout(name="left"), Layout(name="right"))
        lay["left"].split_column(Layout(self.throughput(), size=8), Layout(self.results()))
        lay["right"].split_column(Layout(self.system()), Layout(self.threads()))
        return lay

    def run(self, log_path: Path) -> dict:
        proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.proc_ps = psutil.Process(proc.pid)
        psutil.cpu_percent(percpu=True)
        with open(log_path, "w") as log, Live(self.render(), refresh_per_second=4, screen=False) as live:
            try:
                for line in proc.stdout:
                    log.write(line)
                    self.hb = json.loads(line)
                    if self.hb["type"] == "heartbeat":
                        self.rates.append(self.hb["rounds_per_s"])
                    live.update(self.render())
            except KeyboardInterrupt:
                proc.terminate()
        proc.wait()
        # Plain-text copy of the final frame, e.g. for reports or remote runs.
        snap = Console(width=110, height=36, record=True, file=open(os.devnull, "w"))
        snap.print(self.render())
        log_path.with_suffix(".txt").write_text(snap.export_text())
        return self.hb


def run(sim_args: list[str], headless: bool = False) -> dict:
    if not SIM.exists():
        raise SystemExit(f"{SIM} not found - run `make` first")
    cmd = [str(SIM), *sim_args]
    log_path = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    RUNS.mkdir(exist_ok=True)
    if headless:
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
            hb = {}
            for line in proc.stdout:
                log.write(line)
                hb = json.loads(line)
                print(f"[{hb['t']:7.1f}s] {human(hb['rounds'])} rounds  {human(hb['rounds_per_s'])}/s  "
                      f"A-B {hb['diff_per_round']:+.2f}±{hb['diff_ci95']:.2f}", flush=True)
            proc.wait()
    else:
        hb = Dashboard(cmd).run(log_path)
    print(f"log: {log_path.relative_to(ROOT)}")
    return hb
