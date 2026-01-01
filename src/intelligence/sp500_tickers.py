"""
S&P 500 tickers helpers (compatibility module).

External schedulers may reference this module as `intelligence.sp500_tickers`.
Implementation delegates to SpringEdge's existing DB-layer helpers.
"""

from __future__ import annotations

from typing import Any

from springedge.layers import fetch_sp500_baseline as _fetch_sp500_baseline

DEFAULT_TABLE = "sp500_tickers"


def fetch_sp500_tickers(
    conn: Any,
    *,
    table: str = DEFAULT_TABLE,
    symbol_col: str = "symbol",
    as_of_col: str = "date",
):
    """
    Fetch the S&P 500 ticker universe from a DB-API connection.

    This is a small wrapper around `springedge.layers.fetch_sp500_baseline`.
    """

    return _fetch_sp500_baseline(
        conn, table=table, symbol_col=symbol_col, as_of_col=as_of_col
    )


__all__ = ["DEFAULT_TABLE", "fetch_sp500_tickers"]
