"""Tests for Kalshi fixed-point (`_fp` / `_dollars`) field parsing.

Payloads mirror real API responses captured on 2026-10-24.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.data.data_hub import DataHub
from src.data.kalshi_client import KalshiRestClient
from src.data.kalshi_fields import (
    get_count,
    get_dollars,
    get_order_count,
    get_quantity,
    parse_orderbook_side,
)
from src.data.models import OrderRequest
from src.features.indicators import orderbook_depth_imbalance

# Kalshi returns levels in ascending price order
WS_SNAPSHOT = {
    "market_ticker": "KXBTC15M-26OCT241000-00",
    "market_id": "01a16a5a-7dcf-7ce7-ae98-b83ed9163513",
    "yes_dollars_fp": [["0.0010", "1969652.00"], ["0.4200", "500.00"], ["0.4300", "1796.83"]],
    "no_dollars_fp": [["0.0010", "1993947.00"], ["0.5500", "200.00"], ["0.5600", "3528.76"]],
}

ORDER_FP = {
    "order_id": "abc",
    "client_order_id": "cid",
    "ticker": "KXBTC15M-26OCT241000-00",
    "side": "yes",
    "action": "buy",
    "status": "executed",
    "yes_price_dollars": "0.4300",
    "no_price_dollars": "0.5700",
    "initial_count_fp": "3.00",
    "fill_count_fp": "2.00",
    "remaining_count_fp": "1.00",
}


def _rest_client(response: dict) -> KalshiRestClient:
    client = object.__new__(KalshiRestClient)
    client._request = AsyncMock(return_value=response)  # type: ignore[method-assign]
    return client


class TestFieldHelpers:
    def test_count_prefers_fp(self):
        assert get_count({"fill_count_fp": "2.00", "fill_count": 9}, "fill_count") == 2

    def test_count_falls_back_to_legacy(self):
        assert get_count({"fill_count": 4}, "fill_count") == 4

    def test_count_missing_is_default(self):
        assert get_count({}, "fill_count") == 0

    def test_quantity_keeps_fraction(self):
        assert get_quantity({"delta_fp": "-1.50"}, "delta") == -1.5

    def test_dollars_prefers_dollars_field(self):
        assert get_dollars({"balance_dollars": "3.0000", "balance": 300}, "balance") == Decimal("3")

    def test_dollars_converts_legacy_cents(self):
        assert get_dollars({"yes_price": 43}, "yes_price") == Decimal("0.43")

    def test_order_count_uses_initial_count(self):
        assert get_order_count(ORDER_FP) == 3
        assert get_order_count({"count": 5}) == 5


class TestOrderbookParsing:
    def test_ws_snapshot_sorted_best_first(self):
        levels = parse_orderbook_side(WS_SNAPSHOT, "yes")
        assert [lvl.price_dollars for lvl in levels] == [
            Decimal("0.4300"), Decimal("0.4200"), Decimal("0.0010"),
        ]
        assert levels[0].quantity == 1796.83

    def test_legacy_cents_format(self):
        levels = parse_orderbook_side({"yes": [[42, 10], [43, 5]]}, "yes")
        assert levels[0].price_dollars == Decimal("0.43")
        assert levels[0].quantity == 5

    def test_data_hub_snapshot_and_delta(self):
        hub = SimpleNamespace(_orderbook_cache={})
        DataHub._handle_orderbook_snapshot(
            hub, WS_SNAPSHOT["market_ticker"], WS_SNAPSHOT, datetime.now(timezone.utc)
        )
        ob = hub._orderbook_cache[WS_SNAPSHOT["market_ticker"]]
        assert ob.best_yes_bid == Decimal("0.43")
        assert ob.best_no_bid == Decimal("0.56")
        assert ob.spread == Decimal("0.01")
        assert orderbook_depth_imbalance(ob.yes_levels, ob.no_levels) != 0.0

        # Real delta shape: string delta_fp, dollar price
        delta = {"price_dollars": "0.5600", "delta_fp": "-3528.76", "side": "no"}
        DataHub._handle_orderbook_delta(hub, WS_SNAPSHOT["market_ticker"], delta)
        assert ob.best_no_bid == Decimal("0.55")

        delta = {"price_dollars": "0.4400", "delta_fp": "10.00", "side": "yes"}
        DataHub._handle_orderbook_delta(hub, WS_SNAPSHOT["market_ticker"], delta)
        assert ob.best_yes_bid == Decimal("0.44")

    async def test_rest_orderbook_fp(self):
        client = _rest_client({
            "orderbook_fp": {
                "yes_dollars": [["0.4200", "500.00"], ["0.4300", "1796.83"]],
                "no_dollars": [["0.5600", "3528.76"]],
            }
        })
        ob = await client.get_orderbook("T")
        assert ob.best_yes_bid == Decimal("0.4300")
        assert ob.best_no_bid == Decimal("0.5600")


class TestRestParsing:
    async def test_market_volume_fp(self):
        client = _rest_client({"market": {
            "ticker": "T", "volume_fp": "1535711.99", "volume_24h_fp": "856007.80",
            "open_interest_fp": "496477.47", "yes_bid_dollars": "0.4300",
        }})
        market = await client.get_market("T")
        assert market.volume == 1535711
        assert market.open_interest == 496477
        assert market.yes_bid == Decimal("0.4300")

    async def test_balance_dollars(self):
        client = _rest_client({"balance": 300, "balance_dollars": "3.0000"})
        assert await client.get_balance() == Decimal("3.0000")

    async def test_create_order_response_fp(self):
        client = _rest_client({"order": ORDER_FP})
        resp = await client.create_order(OrderRequest(
            ticker="T", side="yes", action="buy", count=3,
            yes_price_dollars="0.43", client_order_id="cid",
        ))
        assert (resp.count, resp.fill_count, resp.remaining_count) == (3, 2, 1)

    async def test_positions_fp(self):
        client = _rest_client({"market_positions": [{
            "ticker": "T",
            "position_fp": "-2.00",
            "market_exposure_dollars": "1.140000",
            "fees_paid_dollars": "0.040000",
            "total_traded_dollars": "1.140000",
            "realized_pnl_dollars": "0.000000",
        }]})
        (pos,) = await client.get_positions()
        assert pos.position == -2
        assert pos.market_exposure == 114  # cents
        assert pos.fees_paid == Decimal("0.04")


class TestOrderRequest:
    def test_sends_matching_count_fp(self):
        body = OrderRequest(
            ticker="T", side="yes", action="buy", count=1,
            yes_price_dollars="0.43", client_order_id="cid",
        ).to_api_dict()
        assert body["count"] == 1
        assert body["count_fp"] == "1.00"


class TestMarketScanner:
    async def test_scan_uses_close_window_and_keeps_active(self):
        from datetime import timedelta

        from src.config import KalshiConfig
        from src.data.market_scanner import MarketScanner
        from src.data.models import Market

        now = datetime.now(timezone.utc)
        active = Market(
            ticker="KXBTC15M-NOW", status="active",
            open_time=now - timedelta(minutes=5), close_time=now + timedelta(minutes=10),
        )
        future = Market(
            ticker="KXBTC15M-NEXT", status="initialized",
            open_time=now + timedelta(minutes=10), close_time=now + timedelta(minutes=25),
        )
        client = SimpleNamespace(get_markets=AsyncMock(return_value=[future, active]))
        scanner = MarketScanner(client, KalshiConfig(series_ticker="KXBTC15M"))

        found = await scanner.scan()

        assert [m.ticker for m in found] == ["KXBTC15M-NOW"]
        kwargs = client.get_markets.call_args.kwargs
        assert "status" not in kwargs  # status="open" returned [] for active markets
        assert kwargs["min_close_ts"] <= int(now.timestamp()) + 1
        assert kwargs["max_close_ts"] > kwargs["min_close_ts"]
