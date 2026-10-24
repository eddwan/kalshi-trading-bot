"""Helpers for reading Kalshi API fields across the fixed-point migration.

Kalshi moved contract counts to ``*_fp`` fixed-point strings (e.g. ``"3.00"``)
and prices/amounts to ``*_dollars`` strings (e.g. ``"0.4300"``). Legacy integer
fields (counts, cents) are being removed, so every reader prefers the new field
and falls back to the legacy one.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from src.data.models import OrderbookLevel


def _to_decimal(val: Any) -> Decimal | None:
    if val is None or val == "":
        return None
    try:
        return Decimal(str(val))
    except (InvalidOperation, ValueError):
        return None


def get_count(d: dict, name: str, default: int = 0) -> int:
    """Read a contract count: ``{name}_fp`` (string) first, then legacy ``{name}``."""
    val = _to_decimal(d.get(f"{name}_fp"))
    if val is None:
        val = _to_decimal(d.get(name))
    return int(val) if val is not None else default


def get_quantity(d: dict, name: str, default: float = 0.0) -> float:
    """Like ``get_count`` but keeps fractional contracts (orderbook depth, volume)."""
    val = _to_decimal(d.get(f"{name}_fp"))
    if val is None:
        val = _to_decimal(d.get(name))
    return float(val) if val is not None else default


def get_dollars(d: dict, name: str) -> Decimal | None:
    """Read a dollar amount: ``{name}_dollars`` first, then legacy ``{name}`` in cents."""
    val = _to_decimal(d.get(f"{name}_dollars"))
    if val is not None:
        return val
    cents = _to_decimal(d.get(name))
    return cents / 100 if cents is not None else None


def parse_orderbook_side(payload: dict, side: str) -> list[OrderbookLevel]:
    """Parse one side ("yes"/"no") of a REST or WS orderbook payload.

    Accepts ``{side}_dollars_fp`` (WS), ``{side}_dollars`` (REST) and legacy
    ``{side}`` in cents. Kalshi returns levels in ascending price order; the
    result is sorted descending so ``levels[0]`` is the best bid.
    """
    levels: list[OrderbookLevel] = []
    entries = payload.get(f"{side}_dollars_fp") or payload.get(f"{side}_dollars")
    in_cents = False
    if not entries:
        entries = payload.get(side) or []
        in_cents = True
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        price = _to_decimal(entry[0])
        qty = _to_decimal(entry[1])
        if price is None or qty is None or qty <= 0:
            continue
        if in_cents:
            price = price / 100
        levels.append(OrderbookLevel(price_dollars=price, quantity=float(qty)))
    levels.sort(key=lambda lvl: lvl.price_dollars, reverse=True)
    return levels


def get_order_count(order: dict) -> int:
    """Requested size of an order: ``initial_count_fp`` (new) or ``count``/``count_fp``."""
    val = _to_decimal(order.get("initial_count_fp"))
    if val is not None:
        return int(val)
    return get_count(order, "count")
