"""stammtisch CLI:  python -m stammtisch <command> ...

  monitor [jass-sim args]   run the simulator with a live dashboard
  run     [jass-sim args]   same, plain log lines (for ssh / nohup / CI)
  ingest                    parse data/raw/swisslos -> data/processed/swisslos.npz
  validate                  replay all logged human games through the engine
  bench   [--seconds S]     thread-scaling benchmark -> docs/data/scaling.csv
  stats                     dataset statistics -> docs/data/*.csv
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
    elif cmd == "stats":
        from . import reports
        reports.dataset_stats()
    else:
        sys.exit(f"unknown command {cmd!r}\n{__doc__}")


if __name__ == "__main__":
    main(sys.argv[1:])
