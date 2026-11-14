# polymarket-faa-attribution-02 — nurz entry

Forecasts the Yes price of every market in Polymarket's FAA outage
attribution family, one minute at a time, 240 minutes ahead, at the
quantiles 0.05 / 0.25 / 0.5 / 0.75 / 0.95.

## Program

`main.py` — Python 3.13, standard library only, no network, deterministic.
Run by the grader as `python3 /bot/main.py`; locally:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` in the input directory and writes one `<id>.out`
per case: `{"forecasts": [{"id": <conditionId>, "q": [[5 numbers] x 240]}]}`.

## Model

Per market, an adaptive-volatility random walk around the last price:

```
q_tau(h) = clamp(x - k*x*h + z_tau * sigma * sqrt(h), 0, 1)
```

- `x` — price at the checkpoint.
- `k = 0.0002` — slow decay toward zero (long-shots bleed value as the
  favourite consolidates). A measured short-window mean move was tried
  and rejected: it extrapolates noise and costs far more than it saves.
- `sigma` — the market's own per-minute volatility: an EWMA (120-minute
  half-life) of its recent 1-minute moves over the last 6 hours, blended
  75/25 with its whole-history EWMA, floored at 0.0012 and capped at
  0.0060. The estimate is robust (median-absolute based, so a single
  jump does not inflate it) and the cap stops one jumpy spell from
  blowing the bands out.
- Level scaling: `sigma` is multiplied by `2*sqrt(x(1-x))` (floored at
  0.20), because a probability-like price near 0 or 1 cannot move as
  far in one minute as one near 0.5.
- The 25% / 75% quantiles are narrowed by a factor 0.9; the 5% / 95%
  quantiles keep the full normal z-values.
- Quantiles are rounded to 6 decimals and forced non-decreasing after
  rounding. Short histories fall back gracefully: with fewer than two
  prices the sigma floor applies and the drift is zero.

## Local validation (practice cases p00–p05)

Scored against Polymarket's own public 1-minute price history after each
checkpoint (the same grid RULES describes), mean pinball loss x100:

| case | this entry | house baseline |
|------|-----------:|---------------:|
| p00  |   0.424944 |       0.536138 |
| p01  |   0.308035 |       0.380352 |
| p02  |   0.355402 |       0.408408 |
| p03  |   0.274186 |       0.368599 |
| p04  |   0.188911 |       0.298794 |
| p05  |   0.115200 |       0.238232 |
| **total** | **0.267845** | **0.363589** |

Beats the house baseline on every practice case. Runtime: ~0.07 s per
case locally (limit 8 s); output is byte-identical across runs.

## Notes

- The practice is cut from the "Next CEO of NVIDIA?" event, not the FAA
  family, so its score is no guide to the final — but the mechanisms
  (adaptive per-market volatility, level scaling, slow decay) are
  structural, not fitted to one event's numbers, and the tuned values sit
  on a flat plateau (neighbouring parameter choices score 0.268–0.270).
- The final's markets are young (created 13 Nov 2026) and may hold only
  minutes of history at the first checkpoints; the program handles that
  with the sigma floor and the level factor.
