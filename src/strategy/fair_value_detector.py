"""Fair value vs ask: buy when a spot-implied probability beats the Kalshi ask.

Pre-registered rule from scripts/kalshi_lag_study.py (2026-10-26), forward-tested
in paper only:

    fair = Phi( ln(spot / strike) / (sigma * sqrt(time_to_expiry)) )

where sigma is the realized per-second volatility of the spot sampled each cycle
over the trailing ``fair_value_vol_window_s``. Buy YES when fair - yes_ask >= gap,
or NO when (1 - fair) - no_ask >= gap, as a taker, at most once per market, and
hold to settlement.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timezone

import numpy as np
import structlog

from src.config import StrategyConfig
from src.data.models import MarketSnapshot, TradeSignal

logger = structlog.get_logger()

MIN_VOL_SAMPLES = 50  # ~200s of 4s cycles before the first estimate
MIN_ASK = 0.02
MAX_ASK = 0.98
MIN_TIME_TO_EXPIRY_S = 5.0


class FairValueDetector:
    def __init__(self, config: StrategyConfig) -> None:
        self._config = config
        self._spots: dict[str, deque[tuple[float, float]]] = {}
        self._entered: set[str] = set()
        self.last_reason = ""
        self.last_analysis: dict = {}

    def observe(self, asset: str, timestamp: float, spot: float) -> None:
        """Record a spot sample; call once per cycle per asset."""
        window = self._config.fair_value_vol_window_s
        samples = self._spots.setdefault(asset, deque())
        if samples and samples[-1][0] == timestamp:
            return
        samples.append((timestamp, spot))
        while samples and samples[0][0] < timestamp - window:
            samples.popleft()

    def sigma_per_second(self, asset: str) -> float | None:
        samples = self._spots.get(asset)
        if not samples or len(samples) < MIN_VOL_SAMPLES:
            return None
        times = np.array([s[0] for s in samples])
        log_spot = np.log(np.array([s[1] for s in samples]))
        dt = float(np.median(np.diff(times)))
        sigma = float(np.std(np.diff(log_spot))) / math.sqrt(max(dt, 1e-3))
        return sigma if sigma > 0 else None

    @staticmethod
    def fair_probability(spot: float, strike: float, sigma: float, tte: float) -> float:
        z = math.log(spot / strike) / (sigma * math.sqrt(tte))
        return min(max(0.5 * (1 + math.erf(z / math.sqrt(2))), 0.001), 0.999)

    def evaluate(self, asset: str, snapshot: MarketSnapshot) -> TradeSignal | None:
        """Return a taker buy signal, or None (reason in ``last_reason``)."""
        ticker = snapshot.market_ticker
        now = snapshot.timestamp.timestamp()
        self.observe(asset, now, float(snapshot.spot_price))
        self.last_analysis = {}

        if ticker in self._entered:
            self.last_reason = "Fair value: already entered this market"
            return None
        tte = snapshot.time_to_expiry_seconds
        ob = snapshot.orderbook
        if (
            snapshot.strike_price is None or tte <= MIN_TIME_TO_EXPIRY_S
            or ob.best_yes_bid is None or ob.best_no_bid is None
        ):
            self.last_reason = "Fair value: missing strike/book or too close to expiry"
            return None
        sigma = self.sigma_per_second(asset)
        if sigma is None:
            self.last_reason = "Fair value: warming up volatility estimate"
            return None

        fair = self.fair_probability(
            float(snapshot.spot_price), float(snapshot.strike_price), sigma, tte
        )
        ask_yes = 1.0 - float(ob.best_no_bid)
        ask_no = 1.0 - float(ob.best_yes_bid)
        gap = self._config.fair_value_min_gap
        self.last_analysis = {
            "fair": round(fair, 4), "ask_yes": ask_yes, "ask_no": ask_no, "tte": round(tte, 1),
        }

        for side, ask, prob in (("yes", ask_yes, fair), ("no", ask_no, 1.0 - fair)):
            if MIN_ASK <= ask <= MAX_ASK and prob - ask >= gap:
                self._entered.add(ticker)
                fee = math.ceil(0.07 * ask * (1 - ask) * 100) / 100
                logger.info(
                    "fair_value_signal",
                    ticker=ticker, side=side, fair=round(fair, 4), ask=ask,
                    net_edge=round(prob - ask - fee, 4), tte=round(tte, 1),
                )
                return TradeSignal(
                    market_ticker=ticker,
                    side=side,
                    action="buy",
                    raw_edge=prob - ask,
                    net_edge=prob - ask - fee,
                    model_probability=fair,
                    implied_probability=float(snapshot.implied_yes_prob or 0.5),
                    confidence=1.0,
                    suggested_price_dollars=f"{ask:.4f}",
                    timestamp=datetime.now(timezone.utc),
                    signal_type="fair_value",
                    post_only=False,  # taker: cross the spread at the ask
                )

        self.last_reason = (
            f"Fair value: no gap (fair={fair:.3f} ask_yes={ask_yes:.3f} ask_no={ask_no:.3f})"
        )
        return None
