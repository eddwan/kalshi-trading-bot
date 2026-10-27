"""Tests for FairValueDetector (fair value vs ask, paper forward test)."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from src.config import StrategyConfig, load_settings
from src.data.models import MarketSnapshot, Orderbook, OrderbookLevel
from src.risk.position_sizer import PositionSizer
from src.strategy.fair_value_detector import MIN_VOL_SAMPLES, FairValueDetector


def _snapshot(base: MarketSnapshot, *, spot: float, strike: float, yes_bid: str,
              no_bid: str, tte: float = 600.0, offset_s: float = 0.0,
              ticker: str = "KXBTC15M-T") -> MarketSnapshot:
    ob = Orderbook(
        ticker=ticker,
        yes_levels=[OrderbookLevel(price_dollars=Decimal(yes_bid), quantity=10)],
        no_levels=[OrderbookLevel(price_dollars=Decimal(no_bid), quantity=10)],
        timestamp=base.timestamp,
    )
    return base.model_copy(update={
        "market_ticker": ticker,
        "timestamp": base.timestamp + timedelta(seconds=offset_s),
        "spot_price": Decimal(str(spot)),
        "strike_price": Decimal(str(strike)),
        "orderbook": ob,
        "time_to_expiry_seconds": tte,
    })


def _warm(detector: FairValueDetector, base: MarketSnapshot, level: float = 100.0) -> None:
    """Feed alternating spot samples around ``level`` so sigma is known and non-zero."""
    t0 = base.timestamp.timestamp() - 4 * (MIN_VOL_SAMPLES + 5)
    for i in range(MIN_VOL_SAMPLES + 5):
        detector.observe("BTC", t0 + 4 * i, level * (1.0005 if i % 2 else 1.0))


class TestFairProbability:
    def test_at_the_money_is_half(self):
        assert FairValueDetector.fair_probability(100.0, 100.0, 1e-4, 600) == pytest.approx(0.5)

    def test_above_strike_favors_yes(self):
        assert FairValueDetector.fair_probability(101.0, 100.0, 1e-4, 600) > 0.9


class TestFairValueDetector:
    def test_warm_up_blocks_first_cycles(self, sample_snapshot):
        det = FairValueDetector(StrategyConfig(fair_value_exclusive=True))
        snap = _snapshot(sample_snapshot, spot=101, strike=100, yes_bid="0.40", no_bid="0.55")
        assert det.evaluate("BTC", snap) is None
        assert "warming up" in det.last_reason

    def test_buys_yes_as_taker_when_fair_beats_ask(self, sample_snapshot):
        det = FairValueDetector(StrategyConfig(fair_value_exclusive=True))
        _warm(det, sample_snapshot)
        # spot far above strike -> fair ~1; YES ask = 1 - 0.55 = 0.45
        snap = _snapshot(sample_snapshot, spot=101, strike=100, yes_bid="0.40", no_bid="0.55")
        sig = det.evaluate("BTC", snap)
        assert sig is not None
        assert (sig.side, sig.suggested_price_dollars) == ("yes", "0.4500")
        assert sig.post_only is False and sig.signal_type == "fair_value"

    def test_buys_no_when_below_strike(self, sample_snapshot):
        det = FairValueDetector(StrategyConfig(fair_value_exclusive=True))
        _warm(det, sample_snapshot)
        snap = _snapshot(sample_snapshot, spot=99, strike=100, yes_bid="0.40", no_bid="0.55")
        sig = det.evaluate("BTC", snap)
        assert sig is not None and sig.side == "no"
        assert sig.suggested_price_dollars == "0.6000"  # NO ask = 1 - 0.40

    def test_no_trade_when_gap_too_small(self, sample_snapshot):
        det = FairValueDetector(StrategyConfig(fair_value_exclusive=True))
        _warm(det, sample_snapshot)
        # at the money -> fair 0.5; asks 0.52/0.52 -> gap negative
        snap = _snapshot(sample_snapshot, spot=100, strike=100, yes_bid="0.48", no_bid="0.48")
        assert det.evaluate("BTC", snap) is None
        assert "no gap" in det.last_reason

    def test_enters_each_market_once(self, sample_snapshot):
        det = FairValueDetector(StrategyConfig(fair_value_exclusive=True))
        _warm(det, sample_snapshot)
        snap = _snapshot(sample_snapshot, spot=101, strike=100, yes_bid="0.40", no_bid="0.55")
        assert det.evaluate("BTC", snap) is not None
        again = _snapshot(sample_snapshot, spot=101, strike=100, yes_bid="0.40", no_bid="0.55",
                          offset_s=4)
        assert det.evaluate("BTC", again) is None


class TestFairValueProfile:
    @pytest.mark.parametrize("ask", ["0.05", "0.45", "0.90"])
    def test_profile_sizes_one_contract(self, sample_snapshot, ask):
        settings = load_settings("config/settings.fair_value.yaml")
        assert settings.strategy.fair_value_exclusive is True
        assert not settings.strategy.stop_loss_enabled
        det = FairValueDetector(settings.strategy)
        _warm(det, sample_snapshot, level=105.0)
        no_bid = str(Decimal("1") - Decimal(ask))
        snap = _snapshot(sample_snapshot, spot=105, strike=100, yes_bid="0.01", no_bid=no_bid)
        # fair ~1 -> YES at ask
        sig = det.evaluate("BTC", snap)
        assert sig is not None
        sizer = PositionSizer(settings.risk, strategy_config=settings.strategy)
        assert sizer.size(sig, Decimal("30"), Decimal("0")) == 1
