#!/usr/bin/env bash
# Fetch the Swisslos Schieber logs (HSLU DL4G) from a public student mirror.
# Usage permission has been requested from HSLU - see outreach/email_hslu.md.
set -euo pipefail
cd "$(dirname "$0")/.."
dst=data/raw/swisslos
mkdir -p "$dst"
base=https://github.com/fierc3/DL4G-HSLU/raw/main/Data
for i in $(seq -w 1 19); do
  f=jass_game_00$i.txt.gz
  [ -s "$dst/$f" ] || echo "$base/games/$f"
done | xargs -r -P 6 -n 1 sh -c 'curl -sfL -o "'"$dst"'/$(basename "$0")" "$0"'
[ -s "$dst/player_all_stat.json" ] || curl -sfL -o "$dst/player_all_stat.json" "$base/stat/player_all_stat.json"
[ -s "$dst/2018_10_18_trump.csv" ] || curl -sfL -o "$dst/2018_10_18_trump.csv" \
  https://github.com/xXTime-OnXx/dl4g-jass-bot/raw/main/data/2018_10_18_trump.csv
ls -la "$dst"
