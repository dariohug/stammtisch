"""stammtisch CLI:  python -m stammtisch <command> ...

  monitor [jass-sim args]   run the simulator with a live dashboard
  run     [jass-sim args]   same, plain log lines (for ssh / nohup / CI)
  ingest                    parse data/raw/swisslos -> data/processed/swisslos.npz
  validate                  replay all logged human games through the engine
  bench   [--seconds S]     thread-scaling benchmark -> docs/data/scaling.csv
  stats                     dataset statistics -> docs/data/*.csv
  report-data               everything docs/report.tex needs (stats, bench, solver timing, training, arena)
  train card|trump [opts]   imitation learning on the Swisslos logs -> models/
  train distill [--selfplay runs/selfplay.bin --steps N]   fine-tune on the search agent's decisions
  arena [--agents a,b,..] [--deals N] [--slow-deals M]   gauntlet + Elo -> docs/data/leaderboard*.csv
  gate CANDIDATE --vs BASELINE [--deals N]              exit 1 unless the candidate is clearly better
  advise --hand "..." --dealer me|right|partner|left [--trump T --played "..." --pushed yes]
                            live advice: trump call / push, or the card to play
  play [--bots SPEC] [--no-coach] [--rules R]   play a round against the engine, with a coach
  analyze FILE [opts]       review a recorded game (see stammtisch/record.py for the format)
      --who me|team|all  --samples N  --exact K  --md OUT.md  --html OUT.html  --swisslos INDEX (demo)
"""
from __future__ import annotations

import sys


def main(argv: list[str]) -> None:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd in ("monitor", "run"):
        from . import monitor
        monitor.run(rest, headless=cmd == "run")
    elif cmd == "ingest":
        from . import data
        d = data.ingest()
        print(f"{len(d['trump']):,} games -> {data.PROCESSED}")
    elif cmd == "validate":
        from . import data, engine
        d = data.load()
        status, _ = engine.replay_check(d)
        for code, name in enumerate(engine.REPLAY_STATUS):
            print(f"{name:>24}: {int((status == code).sum()):,}")
        sys.exit(int((status != 0).any()))
    elif cmd == "bench":
        from . import reports
        reports.bench(float(rest[rest.index("--seconds") + 1]) if "--seconds" in rest else 3.0)
    elif cmd == "report-data":
        from . import reports
        reports.dataset_stats()
        reports.bench(3.0)
        reports.dd_timing()
        reports.training_curves()
        reports.arena_numbers(rest[0] if rest else reports.DEFAULT_ARENA_NOTE)
    elif cmd == "stats":
        from . import reports
        reports.dataset_stats()
    elif cmd == "analyze":
        from . import analyze, data, record
        opts = dict(zip(rest[1::2], rest[2::2])) if rest and not rest[0].startswith("--") else dict(zip(rest[::2], rest[1::2]))
        if "--swisslos" in opts:
            game = record.from_swisslos(data.load(), int(opts["--swisslos"]))
            print(record.dump(game))
        else:
            game = record.parse(open(rest[0], encoding="utf-8").read())
        a = analyze.analyze(game, samples=int(opts.get("--samples", 64)), exact_left=int(opts.get("--exact", 6)),
                            who=opts.get("--who", "me"))
        analyze.render(a)
        if "--md" in opts:
            open(opts["--md"], "w", encoding="utf-8").write(analyze.to_markdown(a))
        if "--html" in opts:
            open(opts["--html"], "w", encoding="utf-8").write(analyze.to_html(a))
            print(f"HTML review: {opts['--html']}")
    elif cmd == "advise":
        from . import advise
        o = dict(zip(rest[::2], rest[1::2]))
        advise.advise(o["--hand"], o.get("--dealer", "left"), o.get("--trump"),
                      o.get("--pushed", "no").lower() in ("yes", "ja", "1", "true"), o.get("--played", ""),
                      o.get("--rules", "swisslos"), int(o.get("--samples", 64)))
    elif cmd == "play":
        from . import play
        o = dict(zip([x for x in rest if x.startswith("--") and x != "--no-coach"],
                     [rest[i + 1] for i, x in enumerate(rest) if x.startswith("--") and x != "--no-coach"]))
        play.play(bots=o.get("--bots", "pimc:n=24"), rules=o.get("--rules", "swisslos"),
                  coach="--no-coach" not in rest)
    elif cmd == "arena":
        from . import arena
        opts = dict(zip(rest[::2], rest[1::2]))
        agents = opts.get("--agents", ",".join(arena.DEFAULT_AGENTS)).split(",")
        arena.gauntlet(agents, int(opts["--deals"]) if "--deals" in opts else None,
                       int(opts["--slow-deals"]) if "--slow-deals" in opts else None)
    elif cmd == "gate":
        from . import arena
        opts = dict(zip(rest[1::2], rest[2::2]))
        sys.exit(0 if arena.gate(rest[0], opts.get("--vs", "net"), int(opts.get("--deals", 400))) else 1)
    elif cmd == "train":
        from . import train
        opts = dict(zip(rest[1::2], rest[2::2]))
        kw = {k.lstrip("-"): (float(v) if "." in v else int(v)) if v.replace(".", "").isdigit() else v
              for k, v in opts.items()}
        {"card": train.train_card, "trump": train.train_trump, "distill": train.distill}[rest[0]](**kw)
    else:
        sys.exit(f"unknown command {cmd!r}\n{__doc__}")


if __name__ == "__main__":
    main(sys.argv[1:])
