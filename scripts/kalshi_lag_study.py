#!/usr/bin/env python3
"""
Does the Kalshi price lag the spot price?

Uses the raw per-cycle data the bot logs in observe mode (predictions.features_json
"_market": spot, strike, Kalshi bids, time to expiry; ~4s resolution).

1. Lead-lag: does the spot-implied probability move over the last k seconds predict
   the Kalshi mid move over the next k seconds? (positive slope = Kalshi lags)
2. Information: does a spot-based fair value add to the Kalshi mid for predicting
   the outcome, out of sample?
3. Tradability: buy the side where fair value exceeds the Kalshi ASK by a threshold,
   hold to settlement, pay taker fees. P&L per contract with a block-bootstrap CI.

Usage:
    set -a; source .env; set +a
    .venv/bin/python scripts/kalshi_lag_study.py [--since 2026-10-25T16:03]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signal_value_test import fetch_results  # noqa: E402

VOL_WINDOW_S = 900       # trailing window for realized vol
LAGS_S = (8, 20, 60)
THRESHOLDS = (0.05, 0.10, 0.15)


def phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def taker_fee(price: float) -> float:
    return math.ceil(0.07 * price * (1 - price) * 100) / 100


def load(db: str, since: str) -> dict[str, list[dict]]:
    rows = sqlite3.connect(db).execute(
        "select market_ticker, timestamp, json_extract(features_json, '$._market') "
        "from predictions where timestamp >= ? and json_extract(features_json, '$._market') "
        "is not null order by timestamp", (since,),
    ).fetchall()
    results = asyncio.run(fetch_results(sorted({r[0] for r in rows})))
    # realized vol per asset from the logged spot series
    spot_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    parsed = []
    for ticker, ts, raw in rows:
        m = json.loads(raw)
        t = datetime.fromisoformat(ts).timestamp()
        asset = re.match(r"KX([A-Z]+)15M", ticker).group(1)
        if m.get("spot"):
            spot_series[asset].append((t, m["spot"]))
        parsed.append((ticker, asset, t, m))
    vol_idx = {a: (np.array([x[0] for x in s]), np.log(np.array([x[1] for x in s])))
               for a, s in spot_series.items()}

    markets: dict[str, list[dict]] = defaultdict(list)
    for ticker, asset, t, m in parsed:
        res = results.get(ticker)
        yes_bid, no_bid = m.get("yes_bid"), m.get("no_bid")
        if (not res or None in (m.get("spot"), m.get("strike"), yes_bid, no_bid)
                or not 5 < m["tte"] <= 900):
            continue
        ts_arr, lp = vol_idx[asset]
        i = np.searchsorted(ts_arr, t, side="right")
        j = np.searchsorted(ts_arr, t - VOL_WINDOW_S)
        if i - j < 50:
            continue
        rets = np.diff(lp[j:i])
        dt = np.median(np.diff(ts_arr[j:i]))
        sigma_per_s = np.std(rets) / math.sqrt(max(dt, 1e-3))
        if sigma_per_s <= 0:
            continue
        z = math.log(m["spot"] / m["strike"]) / (sigma_per_s * math.sqrt(m["tte"]))
        markets[ticker].append({
            "t": t, "tte": m["tte"], "fair": min(max(phi(z), 0.001), 0.999),
            "mid": (yes_bid + 1 - no_bid) / 2, "ask_yes": 1 - no_bid, "ask_no": 1 - yes_bid,
            "y": 1.0 if res["result"] == "yes" else 0.0,
            "close": datetime.fromisoformat(res["close_time"].replace("Z", "+00:00")).timestamp(),
        })
    return markets


def lead_lag(markets: dict[str, list[dict]]) -> None:
    print("1) Lead-lag: next-k Kalshi move ~ past-k fair-value move (positive = Kalshi lags)")
    for k in LAGS_S:
        past, future = [], []
        for obs in markets.values():
            t = np.array([o["t"] for o in obs])
            lf = logit(np.array([o["fair"] for o in obs]))
            lm = logit(np.array([o["mid"] for o in obs]))
            for i in range(len(obs)):
                a = np.searchsorted(t, t[i] - k)
                b = np.searchsorted(t, t[i] + k)
                if a >= i or b >= len(obs) or abs(t[i] - t[a] - k) > 4 or abs(t[b] - t[i] - k) > 4:
                    continue
                # control for Kalshi's own past move so we measure spot information only
                past.append((lf[i] - lf[a], lm[i] - lm[a]))
                future.append(lm[b] - lm[i])
        x = np.array(past)
        y = np.array(future)
        design = np.column_stack([np.ones(len(x)), x])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        se = np.sqrt(np.diag(np.linalg.inv(design.T @ design)) * resid.var())
        print(f"   k={k:>3}s n={len(y):>6} spot-move coef={coef[1]:+.4f} (t={coef[1] / se[1]:+.1f})"
              f"  own-move coef={coef[2]:+.4f}")


def information(markets: dict[str, list[dict]]) -> None:
    print("\n2) Out of sample: market mid vs mid + spot fair value (log loss, lower = better)")
    order = sorted(markets, key=lambda m: markets[m][0]["close"])
    cut = len(order) // 2

    def stack(names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = [o for n in names for o in markets[n]]
        mid = logit(np.array([o["mid"] for o in obs]))
        fair = logit(np.array([o["fair"] for o in obs]))
        return np.column_stack([np.ones(len(obs)), mid, fair]), np.array([o["y"] for o in obs]), mid

    def fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        w = np.zeros(x.shape[1])
        for _ in range(40):
            p = 1 / (1 + np.exp(-x @ w))
            w -= np.linalg.solve(x.T @ (x * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(w)),
                                 x.T @ (p - y))
        return w

    def ll(p: np.ndarray, y: np.ndarray) -> float:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    xtr, ytr, _ = stack(order[:cut])
    xte, yte, mid_te = stack(order[cut:])
    wa, wb = fit(xtr[:, :2], ytr), fit(xtr, ytr)
    pa = 1 / (1 + np.exp(-xte[:, :2] @ wa))
    pb = 1 / (1 + np.exp(-xte @ wb))
    print(f"   test markets={len(order) - cut}  raw mid={ll(1 / (1 + np.exp(-mid_te)), yte):.4f}"
          f"  mid recal.={ll(pa, yte):.4f}  mid+fair={ll(pb, yte):.4f}  fair weight={wb[2]:+.3f}")


def tradability(markets: dict[str, list[dict]]) -> None:
    print("\n3) Strategy: buy when fair - ask >= threshold, hold to settlement, taker fees")
    rng = np.random.default_rng(0)
    for thr in THRESHOLDS:
        pnl: dict[str, float] = {}
        for name, obs in markets.items():
            for o in obs:
                for side, ask, prob in (("yes", o["ask_yes"], o["fair"]),
                                        ("no", o["ask_no"], 1 - o["fair"])):
                    if 0.02 <= ask <= 0.98 and prob - ask >= thr:
                        won = (o["y"] == 1.0) == (side == "yes")
                        pnl[name] = (1.0 if won else 0.0) - ask - taker_fee(ask)
                        break
                if name in pnl:
                    break  # one entry per market
        if not pnl:
            print(f"   thr={thr:.2f}: no trades")
            continue
        v = np.array(list(pnl.values()))
        boots = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        print(f"   thr={thr:.2f}: trades={len(v):>4} win={np.mean(v > 0):.1%}"
              f" avg P&L/contract={v.mean():+.4f}"
              f"  95% CI [{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--since", default="2026-10-25T16:03")
    parser.add_argument("--db", default="data/bot.db")
    args = parser.parse_args()
    markets = load(args.db, args.since)
    n_obs = sum(len(v) for v in markets.values())
    print(f"markets={len(markets)} observations={n_obs} since={args.since}\n")
    lead_lag(markets)
    information(markets)
    tradability(markets)


if __name__ == "__main__":
    main()
