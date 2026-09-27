#!/usr/bin/env bash
# Nightly regression: build, all tests, full replay of the human games, arena gauntlet.
# Log: runs/nightly/<date>.log ; results: docs/data/leaderboard*.csv
# Install as a user timer (optional):  cp scripts/systemd/stammtisch-nightly.* ~/.config/systemd/user/
#                                      systemctl --user enable --now stammtisch-nightly.timer
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/nightly
log=runs/nightly/$(date +%F).log
exec > >(tee -a "$log") 2>&1
echo "== $(date) $(git rev-parse --short HEAD 2>/dev/null || echo no-git)"
make engine
build/engine-test
.venv/bin/python -m pytest -q tests
[ -f data/processed/swisslos.npz ] && .venv/bin/python -m stammtisch validate
.venv/bin/python -m stammtisch arena
echo "== done $(date)"
