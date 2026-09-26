# Data

`raw/` and `processed/` are gitignored. Nothing in this folder gets committed.

## Swisslos Schieber logs (HSLU DL4G)

- **Content:** 1'821'878 complete Schieber games played on swisslos.ch between 2017-10-08 and 2018-04-15.
  Each game has all 9 tricks (cards in play order, first seat, winner, points), plus trump, dealer, a push flag and player ids.
  All four hands can be reconstructed from the tricks.
- **Trump file:** `2018_10_18_trump.csv` holds 359'825 trump decisions (36 hand bits, player id, trump; 6 = push).
- **Origin:** Prepared by HSLU (Prof. Thomas Koller) for the *Deep Learning for Games* module.
  `scripts/fetch_data.sh` downloads it from a public student mirror (`fierc3/DL4G-HSLU`, `xXTime-OnXx/dl4g-jass-bot`).
- **License / permission:** Unclear. A request is drafted in `outreach/email_hslu.md`.
  Until HSLU answers, use the data for local experiments only. Do not redistribute it or publish derived datasets.

```bash
make data                         # fetch + convert -> processed/swisslos.npz (~25 s)
python -m stammtisch validate     # replay every game through the engine
```

`processed/swisslos.npz` arrays: `cards[N,36]` (play order), `first/win[N,9]`, `points[N,9]`,
`trump`, `dealer`, `forehand` (1 = forehand declared, 0 = pushed), `player_ids[N,4]`, `date` (unix seconds).
