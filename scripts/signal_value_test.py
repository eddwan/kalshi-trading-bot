#!/usr/bin/env python3
"""
Do the model's signals add information beyond the Kalshi market price?

Joins logged predictions (data/bot.db) with official Kalshi results and runs an
out-of-sample logistic regression with rolling-origin folds (train on earlier
markets, test on the next block):

    A: outcome ~ logit(market price) + logit(price) x time elapsed   (baseline)
    B: A + the model's signals (L2-regularized; market terms unpenalized)

If B never beats A with a 95% block-bootstrap CI that excludes 0, the signals
carry no usable edge over the price. Also reports the heuristic model vs price.

Usage:
    set -a; source .env; set +a
    .venv/bin/python scripts/signal_value_test.py [--since 2026-10-24T14:05] [--db data/bot.db]

Needs KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH (read-only market lookups).
Results are cached in data/market_results.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.kalshi_auth import KalshiAuth  # noqa: E402

BASE_URL = "https://api.elections.kalshi.com"
CACHE = Path("data/market_results.json")
SIGNALS = [
    "mom_signal", "tech_signal", "flow_signal", "mr_signal", "cross_exchange_signal",
    "taker_signal", "settlement_signal", "cross_asset_signal", "chainlink_signal",
    "btc_beta_signal", "hour_signal", "funding_signal", "liquidation_signal",
    "funding_divergence_signal", "liquidation_ratio_signal", "ppe_signal",
    "time_decay_signal", "raw_adjustment",
]
FOLDS = ((0.5, 0.67), (0.67, 0.83), (0.83, 1.0))


async def fetch_results(tickers: list[str]) -> dict[str, dict]:
    """Official result + close time per ticker (cached; only finalized ones are kept)."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    missing = [t for t in tickers if t not in cache]
    if missing:
        auth = KalshiAuth(os.environ["KALSHI_API_KEY_ID"], os.environ["KALSHI_PRIVATE_KEY_PATH"])
        sem = asyncio.Semaphore(5)
        async with aiohttp.ClientSession() as session:

            async def one(ticker: str) -> None:
                path = f"/trade-api/v2/markets/{ticker}"
                async with sem, session.get(
                    BASE_URL + path, headers=auth.get_headers("GET", path)
                ) as resp:
                    market = (await resp.json()).get("market", {})
                if market.get("result") in ("yes", "no"):
                    cache[ticker] = {
                        "result": market["result"], "close_time": market["close_time"],
                    }

            await asyncio.gather(*(one(t) for t in missing))
        CACHE.write_text(json.dumps(cache))
    return cache


def load_rows(db_path: str, since: str) -> list[tuple]:
    rows = sqlite3.connect(db_path).execute(
        "select market_ticker, model_probability, implied_probability, features_json, timestamp "
        "from predictions where timestamp >= ?", (since,),
    ).fetchall()
    results = asyncio.run(fetch_results(sorted({r[0] for r in rows})))
    data = []
    for ticker, model_p, market_p, features_json, ts in rows:
        res = results.get(ticker)
        if not res or not features_json or not 0 < market_p < 1:
            continue
        close = datetime.fromisoformat(res["close_time"].replace("Z", "+00:00"))
        time_left = (close - datetime.fromisoformat(ts)).total_seconds()
        if not 0 < time_left <= 900:
            continue
        features = json.loads(features_json)
        signals = [float(features.get(k, 0) or 0) for k in SIGNALS]
        data.append((ticker, close, time_left, market_p, model_p, signals,
                     1.0 if res["result"] == "yes" else 0.0))
    return data


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def design(rows: list[tuple], with_signals: bool) -> tuple[np.ndarray, np.ndarray]:
    market = logit(np.array([r[3] for r in rows]))
    elapsed = 1 - np.array([r[2] for r in rows]) / 900
    cols = [np.ones(len(rows)), market, market * elapsed]
    if with_signals:
        cols += list(np.array([r[5] for r in rows]).T)
    return np.column_stack(cols), np.array([r[6] for r in rows])


def fit(x: np.ndarray, y: np.ndarray, lam: float) -> tuple:
    """Newton-Raphson logistic regression; intercept and market terms unpenalized."""
    mu, sd = x[:, 1:].mean(0), x[:, 1:].std(0) + 1e-9
    z = x.copy()
    z[:, 1:] = (x[:, 1:] - mu) / sd
    penalty = lam * np.eye(x.shape[1])
    penalty[:3, :3] = 0
    w = np.zeros(x.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-z @ w))
        hess = z.T @ (z * (p * (1 - p))[:, None]) + penalty + 1e-6 * np.eye(len(w))
        w -= np.linalg.solve(hess, z.T @ (p - y) + penalty @ w)
    return w, mu, sd


def predict(model: tuple, x: np.ndarray) -> np.ndarray:
    w, mu, sd = model
    z = x.copy()
    z[:, 1:] = (x[:, 1:] - mu) / sd
    return np.clip(1 / (1 + np.exp(-z @ w)), 1e-6, 1 - 1e-6)


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def bootstrap_ci(rows: list[tuple], p_new: np.ndarray, p_base: np.ndarray,
                 y: np.ndarray, n: int = 500) -> tuple[float, float]:
    """95% CI of log-loss difference, resampling whole markets."""
    tickers = np.array([r[0] for r in rows])
    markets = np.unique(tickers)
    index = {m: np.where(tickers == m)[0] for m in markets}
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(n):
        s = np.concatenate([index[m] for m in rng.choice(markets, len(markets))])
        diffs.append(log_loss(p_new[s], y[s]) - log_loss(p_base[s], y[s]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--since", default="2026-10-24T14:05",
                        help="first prediction timestamp (default: after the orderbook fix)")
    parser.add_argument("--db", default="data/bot.db")
    args = parser.parse_args()

    data = load_rows(args.db, args.since)
    markets = sorted({(r[1], r[0]) for r in data})
    print(f"rows={len(data)} markets={len(markets)} since={args.since}\n")
    helped = 0
    for i, (start, end) in enumerate(FOLDS, 1):
        train_m = {m for _, m in markets[: int(len(markets) * start)]}
        test_m = {m for _, m in markets[int(len(markets) * start): int(len(markets) * end)]}
        train = [r for r in data if r[0] in train_m]
        test = [r for r in data if r[0] in test_m]
        xa, y = design(train, False)
        xa_t, y_t = design(test, False)
        p_base = predict(fit(xa, y, 0.0), xa_t)
        p_market = np.clip(np.array([r[3] for r in test]), 1e-6, 1 - 1e-6)
        p_model = np.clip(np.array([r[4] for r in test]), 1e-6, 1 - 1e-6)
        print(f"fold {i}: train={len(train_m)} test={len(test_m)} markets")
        print(f"  market price      logloss={log_loss(p_market, y_t):.4f}")
        print(f"  A market recal.   logloss={log_loss(p_base, y_t):.4f}")
        print(f"  heuristic model   logloss={log_loss(p_model, y_t):.4f}")
        xb, _ = design(train, True)
        xb_t, _ = design(test, True)
        for lam in (1e2, 1e4):
            p_sig = predict(fit(xb, y, lam), xb_t)
            lo, hi = bootstrap_ci(test, p_sig, p_base, y_t)
            verdict = "HELPS" if hi < 0 else ("hurts" if lo > 0 else "no evidence")
            helped += hi < 0
            delta = log_loss(p_sig, y_t) - log_loss(p_base, y_t)
            print(f"  B +signals lam={lam:.0e} delta={delta:+.4f}"
                  f"  95% CI [{lo:+.4f}, {hi:+.4f}]  {verdict}")
        print()
    print("Conclusion:", "signals add information in some folds — investigate"
          if helped else "no evidence the signals beat the market price")


if __name__ == "__main__":
    main()
