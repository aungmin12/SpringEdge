from __future__ import annotations

from enum import StrEnum


class StockUniverse(StrEnum):
    """
    High-level universe tags. You can still pass arbitrary strings to the
    feature pipeline; this enum is just a convenient set of defaults.
    """

    SP500 = "sp500"
    PENNY = "penny"
    OTHER = "other"


DEFAULT_HORIZON_DAYS: tuple[int, ...] = (7, 21, 30, 60, 90, 180, 365)


def calendar_days_to_trading_days(days: int, *, trading_days_per_year: int = 252) -> int:
    """
    Convert calendar days to approximate trading days.

    Uses a simple linear scaling: trading ≈ round(days * 252 / 365).
    """

    if days <= 0:
        raise ValueError("days must be positive")
    return int(round(days * trading_days_per_year / 365.0))

