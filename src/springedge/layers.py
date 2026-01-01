from __future__ import annotations

from enum import StrEnum
import re
from datetime import date, datetime
from typing import Any

import pandas as pd


class StockUniverse(StrEnum):
    """
    High-level universe tags. You can still pass arbitrary strings to the
    feature pipeline; this enum is just a convenient set of defaults.
    """

    SP500 = "sp500"
    PENNY = "penny"
    OTHER = "other"


DEFAULT_HORIZON_DAYS: tuple[int, ...] = (7, 21, 30, 60, 90, 180, 365)


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_ident(name: str, *, kind: str) -> str:
    """
    Validate a SQL identifier (table/column) to reduce injection risk.

    This package intentionally stays DB-API lightweight (no SQLAlchemy),
    so we validate identifiers and embed them directly into SQL strings.
    """
    n = str(name or "").strip()
    if not n or not _IDENT_RE.fullmatch(n):
        raise ValueError(f"Invalid {kind} identifier: {name!r}")
    return n


def _as_iso_date(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    # Coerce string to a date-like ISO format (YYYY-MM-DD).
    ts = pd.to_datetime(value, errors="raise")
    return ts.date().isoformat()


def fetch_sp500_baseline(
    conn: Any,
    *,
    table: str = "sp500",
    symbol_col: str = "symbol",
    as_of_col: str | None = "date",
    as_of: str | date | datetime | None = None,
) -> pd.DataFrame:
    """
    Fetch the S&P 500 membership from the DB as the baseline universe.

    Assumptions / defaults:
    - Table name defaults to `sp500`
    - Symbol column defaults to `symbol`
    - If `as_of_col` is provided (default: `date`), the baseline is the latest
      snapshot in that table (MAX(date)), or the latest snapshot <= `as_of`.

    Returns a DataFrame with a single column named `symbol_col`, sorted and
    de-duplicated.
    """
    t = _validate_ident(table, kind="table")
    sym = _validate_ident(symbol_col, kind="column")
    asofc = _validate_ident(as_of_col, kind="column") if as_of_col else None

    if asofc is None:
        sql = f"SELECT DISTINCT {sym} AS {sym} FROM {t}"
    else:
        if as_of is None:
            sql = f"""
            SELECT {sym} AS {sym}
            FROM {t}
            WHERE {asofc} = (SELECT MAX({asofc}) FROM {t})
            """
        else:
            as_of_iso = _as_iso_date(as_of)
            sql = f"""
            SELECT {sym} AS {sym}
            FROM {t}
            WHERE {asofc} = (
              SELECT MAX({asofc}) FROM {t} WHERE {asofc} <= '{as_of_iso}'
            )
            """

    df = pd.read_sql_query(sql, conn)
    if sym not in df.columns:
        # Defensive: if a driver returns uppercased names, fall back to first column.
        df.columns = [str(c) for c in df.columns]
        if sym not in df.columns and len(df.columns) == 1:
            df = df.rename(columns={df.columns[0]: sym})

    out = df[[sym]].copy()
    out[sym] = out[sym].astype("string")
    out = out.dropna().drop_duplicates().sort_values(sym).reset_index(drop=True)
    return out


def calendar_days_to_trading_days(days: int, *, trading_days_per_year: int = 252) -> int:
    """
    Convert calendar days to approximate trading days.

    Uses a simple linear scaling: trading ≈ round(days * 252 / 365).
    """

    if days <= 0:
        raise ValueError("days must be positive")
    return int(round(days * trading_days_per_year / 365.0))

