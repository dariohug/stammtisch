# stammtisch

An AI for **Schieber**, the classic Swiss Jass variant, and a coach for your own games.
The progress log lives in [`docs/report.pdf`](docs/report.pdf), which covers the plan, data, engine, agents, arena and validation.

## Quick start

```bash
make venv          # .venv with numpy, rich, psutil, pytest, jass-kit (reference impl.); torch for training
make               # build the C++ engine -> build/jass-sim, build/libjass.so, build/engine-test
make data          # fetch Swisslos logs (see data/README.md) -> data/processed/swisslos.npz
make test          # engine self-tests + Python tests (rules, jass-kit diff, replay of 1.8 M human games)
make train         # imitation networks (card + trump policy) -> models/
make arena         # duplicate-deal gauntlet + Elo -> docs/data/leaderboard.csv
```

## Review your own games

Write a game down in the text format below, then run the review:

```bash
python -m stammtisch analyze my_game.jass            # your decisions
python -m stammtisch analyze my_game.jass --who team # you and your partner
python -m stammtisch analyze my_game.jass --md review.md --samples 128
```

```text
# Stammtisch, Runde 3
players: Dario Anna Beat Chris    # playing order (counter-clockwise), partners sit opposite
me: Dario
dealer: Chris
trump: Rosen                       # Eichel, Rosen, Schilten, Schellen, Obenabe, Undeufe (or Ecken/Herz/…)
pushed: no
rules: swisslos                    # Weis, Stöck, x1/x2/x3 — or plain
tricks:                            # leader first; "name:" labels are optional but get checked
  Anna: EiA  Beat: Ei6  Chris: EiK  Dario: RoU
  Dario: RoA  Anna: Ro6  Beat: Ro7  Chris: Ro9
  ...
```

- **Card names:** German (`RoA`, `SchiU`, `Ei9`, `ScheB`) or French (`HA`, `♠J`, `D9`, `C10`).
  Ober = Q, Under = J, Banner = 10.
- **Partial games:** if you only noted some tricks, add a `hands:` block with all four hands.
- **Example:** [`examples/example_game.jass`](examples/example_game.jass).

### At the table and for practice

```bash
python -m stammtisch advise --hand "RoA RoK RoU Ro9 Ro6 SchiA Schi10 Ei7 Sche6" --dealer left   # trump call / push?
python -m stammtisch advise --hand "..." --dealer left --trump Rosen --played "EiA Ei6 Ei8"      # which card?
python -m stammtisch play                # play a round vs. the engine; the coach comments on every decision
```

`play` saves every round to `runs/games/*.jass`, so the full review is one command away.

**What the review does.** For every decision (the trump call too), it estimates how many card points each option was worth.

- **What you knew:** the unseen cards are sampled consistently with everything you could see, including your hand, the played cards, voids and shown Weis. The samples are weighted by what the others' trump call and card play revealed.
- **How a sample is scored:** each one is played out by a human-like policy, then solved exactly for the last tricks.
- **Verdicts:** ✓ good, `?!` inaccuracy (≥ 1 point), `?` mistake (≥ 4), `??` blunder (≥ 10).
  Every loss comes with a ± standard error. A decision only counts as a clear mistake when the loss exceeds twice that noise; `~` marks the unclear ones, and `--samples 256` sharpens them.
- **Reasons:** each mistake gets a one-line reason taken from the actual trick, e.g. "schmieren: RoB gives your team 10 more points".
- **Hindsight:** a separate column shows the best card with all cards open. It's exact, and only appears in the last tricks.

## Agents

| spec | what it is |
|---|---|
| `random` | uniformly random legal moves |
| `heuristic` | hand-written rules (DL4G trump score, schmieren, cheapest winner) |
| `net` / `net:sample` | imitation of 65 M human decisions (Swisslos), argmax / sampled |
| `pimc[:n=32,t=0,k=5,b=0.5,c=4,roll=net]` | search: n sampled deals per card, t per trump call (0 = trump policy), exact solving for the last k tricks, rollouts by `roll`, belief weighting b with c candidates per sample |

```bash
build/jass-sim --a pimc --b net --games 500          # any two agents, duplicate deals, JSONL
python -m stammtisch monitor --a pimc --b net --games 500   # live dashboard
python -m stammtisch gate "pimc:n=64" --vs pimc --deals 400 # exit 1 unless clearly better
```

### Inside HSLU jass-kit

```python
from stammtisch.jasskit_agent import StammtischAgent   # a jass-kit Agent (arena, player service, DL4G tournaments)
agent = StammtischAgent("pimc")
```

## Scheduled tests

- **CI:** `.github/workflows/ci.yml` runs on every push and weekly. It builds the engine, runs the solver self-tests (exact vs. brute force), the Python tests and a baseline arena.
- **Nightly:** `scripts/nightly.sh` rebuilds, runs all tests and the replay of all human games, then plays the arena gauntlet. It appends to `docs/data/leaderboard_history.csv`, and a user-level systemd timer is in `scripts/systemd/`.

## Layout

| path | what |
|---|---|
| `engine/` | C++17: rules + Weis/Stöck (bitboards), double-dummy solver, PIMC search, belief sampling, MLP inference, agents, simulator, C API, self-tests |
| `stammtisch/` | Python: ctypes binding, data ingest, training, game records + review, arena, monitor, report numbers |
| `tests/` | rule tests, differential tests vs. HSLU `jass-kit`, Swisslos replay, record/review tests |
| `docs/` | LaTeX progress log (`make report`) |
| `models/` | trained networks (not in git: derived from the Swisslos data, licence pending) |
| `outreach/` | the data request to HSLU |

Card, trump and seat encoding is identical to HSLU `jass-kit` (card = suit·9 + rank, suits D/H/S/C, ranks A K Q J 10 9 8 7 6).
German suit names follow Swisslos: Eichel = ♦, Rosen = ♥, Schilten = ♠, Schellen = ♣.
