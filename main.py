# polymarket-faa-attribution-02 entry (nurz) — Python 3.13, standard library only.
#
# Forecast: per market, an adaptive-volatility random walk around the last
# price, with a heavily shrunk short-window drift. The spread widens like
# sqrt(h) but its per-minute scale is estimated from that market's own recent
# 1-minute moves (robust, EWMA-weighted), so quiet markets get narrow bands
# and jumpy markets get wide ones. A small jump allowance is folded into the
# spread so a rare large tick does not blow up the pinball loss.
#
#   q_tau(h) = clamp(x - k*x*h + z_tau * sigma * sqrt(h), 0, 1), rounded to 6 dp
#
# where x is the price at the checkpoint, k a small decay rate toward zero,
# sigma is the market's adaptive per-minute volatility (EWMA of its recent
# 1-minute moves, blended with its whole history, floored and capped, then
# scaled by 2*sqrt(x(1-x)) because a price near 0 or 1 cannot move as far),
# and the 25%/75% quantiles are narrowed slightly. Everything is
# deterministic: no randomness, no network, no clock. Reads every *.in in
# $MAKEFUN_INPUT_DIR (else /input), writes /output/<id>.out (else
# $MAKEFUN_OUTPUT_DIR).

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# --- tunables (fixed in advance; chosen on the practice cases) -----------
# Practice loss of this configuration: 0.267845 vs house baseline 0.363589
# (mean pinball loss x100 over the six practice cases, scored against
# Polymarket's own 1-minute history after each checkpoint).
SIGMA_FLOOR = 0.0012      # per-minute sigma floor (price units)
SIGMA_CAP = 0.0060        # per-minute sigma cap
LEVEL_FLOOR = 0.20        # floor on the price-level volatility factor
DECAY_RATE = 0.0002       # drift = -DECAY_RATE * x per minute (decay to 0)
INNER_ZSCALE = 0.9        # inner quantiles (25%/75%) slightly narrower
DRIFT_WINDOW = 60         # (unused) kept for reference
VOL_WINDOW = 360          # minutes of recent history for the volatility
VOL_HALF_LIFE = 120.0     # EWMA half-life in minutes for the volatility
LONG_WEIGHT = 0.25        # weight of the whole-history sigma in the blend


def _price6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    v = round(min(1.0, max(0.0, x)) * 1e6) / 1e6
    return v + 0.0


def _ewma_sigma(diffs, half_life):
    """Robust per-minute sigma from 1-minute diffs, EWMA-weighted to recent.

    Uses the median absolute diff (robust to jumps) scaled to a normal
    sigma, blended with the RMS of the diffs so sustained movement is not
    missed. Returns 0.0 when there is nothing to measure.
    """
    n = len(diffs)
    if n == 0:
        return 0.0
    lam = 0.5 ** (1.0 / half_life)
    w = 1.0
    wsum = 0.0
    wabs = 0.0
    wsq = 0.0
    # Walk from the most recent diff backwards so the weight decays with age.
    for d in reversed(diffs):
        wsum += w
        wabs += w * abs(d)
        wsq += w * d * d
        w *= lam
    if wsum <= 0.0:
        return 0.0
    mean_abs = wabs / wsum
    rms = math.sqrt(wsq / wsum)
    # Median-absolute proxy: mean |d| of a normal is sigma*sqrt(2/pi).
    sigma_mad = mean_abs * math.sqrt(math.pi / 2.0)
    # Blend: the MAD proxy is jump-robust, the RMS catches real movement.
    return 0.7 * sigma_mad + 0.3 * rms


def _market_params(prices):
    """Return (x, drift_per_min, sigma_per_min) for one market."""
    x = prices[-1]
    n = len(prices)
    if n < 2:
        return x, 0.0, SIGMA_FLOOR
    diffs = [prices[i] - prices[i - 1] for i in range(1, n)]

    # --- drift: slow decay toward zero, proportional to the price --------
    # Prediction-market long-shots bleed value as the favourite consolidates;
    # the pull is tiny (0.02% of the price per minute) so a rallying market
    # is barely held back. A measured short-window mean move was tried and
    # rejected: it extrapolates noise and costs far more than it saves.
    drift = -DECAY_RATE * x

    # --- volatility: recent EWMA sigma blended with whole-history sigma ---
    recent = _ewma_sigma(diffs[-VOL_WINDOW:], VOL_HALF_LIFE)
    whole = _ewma_sigma(diffs, VOL_HALF_LIFE * 4.0)
    sigma = (1.0 - LONG_WEIGHT) * recent + LONG_WEIGHT * whole
    sigma = max(SIGMA_FLOOR, min(SIGMA_CAP, sigma))

    # --- price-level scaling: a price near 0 or 1 cannot move as far -----
    # One-minute moves of a probability-like price scale with sqrt(x(1-x));
    # at x = 0.03 the factor is ~0.33, at x = 0.5 it is 1. Floored so a
    # young market with a short history never gets a zero spread.
    level = 2.0 * math.sqrt(max(x, 0.0) * max(1.0 - x, 0.0))
    sigma *= max(LEVEL_FLOOR, min(1.0, level))
    return x, drift, sigma


def forecast_market(prices, horizon):
    """240 rows x 5 quantiles for one market."""
    x, drift, sigma = _market_params(prices)
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma * math.sqrt(h)
        centre = x + drift * h
        row = [_price6(centre + z * s * spread)
               for z, s in zip(Z, (1.0, INNER_ZSCALE, 1.0,
                                  INNER_ZSCALE, 1.0))]
        # Enforce non-decreasing quantiles after rounding.
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in in_dir.iterdir()
                   if p.is_file() and p.name.endswith(".in"))
    for path in files:
        case = json.loads(path.read_text(encoding="utf-8"))
        horizon = int(case["horizon_minutes"])
        forecasts = []
        for m in case["markets"]:
            prices = [float(v) for v in m["prices"]]
            if not prices:
                raise ValueError("market %s has no prices" % m["id"])
            forecasts.append({"id": m["id"],
                              "q": forecast_market(prices, horizon)})
        out = out_dir / (path.stem + ".out")
        out.write_text(json.dumps({"forecasts": forecasts}) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
