# stammtisch

An AI for **Schieber**, the classic Swiss Jass variant. The progress log lives in
[`docs/report.pdf`](docs/report.pdf), which covers the plan, data, simulator and validation.

## Quick start

```bash
make venv          # .venv with numpy, rich, psutil, pytest, jass-kit (reference impl.)
make               # build the C++ engine -> build/jass-sim, build/libjass.so
make data          # fetch Swisslos logs (see data/README.md) -> data/processed/swisslos.npz
make test          # unit tests, jass-kit differential tests, replay of all human games
make monitor ARGS="--seconds 60 --a heuristic --b random"
```

## Simulator

`build/jass-sim` plays self-play rounds on all cores. Each deal is played twice with the seats
swapped (duplicate), so luck of the cards cancels out. It prints JSON lines: a heartbeat every
`--interval` seconds, then a summary at the end.

```bash
build/jass-sim --games 1000000 --a heuristic --b random      # raw JSONL on stdout
python -m stammtisch monitor --seconds 60                     # live dashboard
python -m stammtisch run --seconds 600                        # plain log lines (ssh, nohup)
```

The dashboard shows throughput, results with 95 % CI, per-core CPU load, frequency, temperature
and per-worker load. Every run is logged to `runs/<timestamp>.jsonl`, and the final frame is
saved to `runs/<timestamp>.txt`.

For learning agents, `stammtisch.engine.VecEnv` steps N rounds per call through the C API.
The action space is: cards 0–35, trump 36–41, push 42.

## Layout

| path | what |
|---|---|
| `engine/` | C++17 rules engine (bitboards), agents, simulator, C API |
| `stammtisch/` | Python: ctypes binding, data ingest, monitor, report numbers |
| `tests/` | rule unit tests, differential tests vs. HSLU `jass-kit`, Swisslos replay |
| `docs/` | LaTeX progress log (`make report`, compiled with tectonic) |
| `outreach/` | correspondence, e.g. the data request to HSLU |

Card, trump and seat encoding is identical to HSLU `jass-kit`
(card = suit·9 + rank, suits D/H/S/C, ranks A K Q J 10 9 8 7 6).
