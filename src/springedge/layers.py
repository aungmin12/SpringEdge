from __future__ import annotations

try:
    # Python 3.11+
    from enum import StrEnum
except ImportError:  # pragma: no cover (only hit on Python < 3.11)
    from enum import Enum

    class StrEnum(str, Enum):
        """
        Backport of `enum.StrEnum` for Python < 3.11.

        This is sufficient for our use-case (string-valued enums for stable tags).
        """

        def __str__(self) -> str:  # matches stdlib behavior
            return str(self.value)

        @classmethod
        def _missing_(cls, value: object) -> "StrEnum | None":
            # Allow construction from raw string values: StockUniverse("sp500")
            if isinstance(value, str):
                for m in cls:
                    if m.value == value:
                        return m
            return None


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


def _validate_table_ref(name: str, *, kind: str = "table") -> str:
    """
    Validate a table reference, optionally schema-qualified.

    Accepts either:
    - table
    - schema.table

    We validate each identifier part to reduce injection risk.
    """
    n = str(name or "").strip()
    if not n:
        raise ValueError(f"Invalid {kind} identifier: {name!r}")
    parts = n.split(".")
    if len(parts) == 1:
        return _validate_ident(parts[0], kind=kind)
    if len(parts) == 2:
        schema = _validate_ident(parts[0], kind="schema")
        table = _validate_ident(parts[1], kind=kind)
        return f"{schema}.{table}"
    raise ValueError(f"Invalid {kind} identifier: {name!r}")


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

    def _safe_rollback() -> None:
        """
        psycopg/psycopg2 can leave the connection in an aborted transaction state
        after an error (e.g. missing table). Rolling back makes subsequent
        fallback queries safe.
        """
        try:
            if hasattr(conn, "rollback"):
                conn.rollback()
        except Exception:
            pass

    def _fetch_df(sql: str) -> pd.DataFrame:
        """
        Execute a SQL query via DB-API cursor and return a DataFrame.

        We intentionally avoid `pandas.read_sql_query` here because it emits
        warnings for some DB-API connections (notably psycopg v3) unless
        SQLAlchemy is used.
        """
        cur = conn.cursor()
        try:
            try:
                cur.execute(sql)
            except Exception:
                # psycopg aborts the current transaction on errors (e.g. missing table).
                # Roll back so callers can safely retry/fallback using the same connection.
                _safe_rollback()
                raise
            rows = cur.fetchall()
            cols = [d[0] for d in (cur.description or [])]
        finally:
            try:
                cur.close()
            except Exception:
                pass
        return pd.DataFrame.from_records(rows, columns=cols)

    def _missing_table_error(err: Exception, *, table_name: str) -> bool:
        """
        Best-effort detection of missing table errors across DB drivers.
        - Postgres (psycopg/psycopg2): relation "x" does not exist
        - sqlite: no such table: x
        """
        msg = str(err).lower()
        tn = str(table_name).lower()
        candidates = {tn}
        if "." in tn:
            candidates.add(tn.split(".")[-1])
        return any(
            (f'relation "{c}" does not exist' in msg) or (f"no such table: {c}" in msg)
            for c in candidates
        )

    def _missing_column_error(err: Exception, *, column_name: str) -> bool:
        """
        Best-effort detection of missing column errors across DB drivers.
        - Postgres: column "x" does not exist
        - sqlite: no such column: x
        """
        msg = str(err).lower()
        cn = str(column_name).lower()
        return (f'column "{cn}" does not exist' in msg) or (
            f"no such column: {cn}" in msg
        )

    def _run_query(
        *, _table: str, _symbol_col: str, _as_of_col: str | None
    ) -> pd.DataFrame:
        t = _validate_table_ref(_table, kind="table")
        sym = _validate_ident(_symbol_col, kind="column")
        asofc = _validate_ident(_as_of_col, kind="column") if _as_of_col else None

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

        df = _fetch_df(sql)
        if sym not in df.columns:
            # Defensive: if a driver returns uppercased names, fall back to first column.
            df.columns = [str(c) for c in df.columns]
            if sym not in df.columns and len(df.columns) == 1:
                df = df.rename(columns={df.columns[0]: sym})

        out = df[[sym]].copy()
        out[sym] = out[sym].astype("string")
        out = out.dropna().drop_duplicates().sort_values(sym).reset_index(drop=True)
        return out

    # Primary attempt (keeps current defaults/tests working).
    try:
        return _run_query(_table=table, _symbol_col=symbol_col, _as_of_col=as_of_col)
    except Exception as exc:
        # Belt-and-suspenders: ensure the connection is usable for fallbacks even
        # if a driver aborted the transaction on the error above.
        _safe_rollback()
        # Production schema compatibility:
        # - Some DBs use `sp500_tickers` instead of `sp500`.
        # - Some DBs don't have an as-of snapshot column; accept a "flat list".
        if table == "sp500" and _missing_table_error(exc, table_name="sp500"):
            # Try common production conventions (with/without schema).
            candidates = ("sp500_tickers", "core.sp500_tickers")
            last_exc: Exception | None = None
            for cand in candidates:
                try:
                    return _run_query(
                        _table=cand, _symbol_col=symbol_col, _as_of_col=as_of_col
                    )
                except Exception as exc2:
                    last_exc = exc2
                    _safe_rollback()
                    if as_of_col is not None and _missing_column_error(
                        exc2, column_name=as_of_col
                    ):
                        return _run_query(
                            _table=cand, _symbol_col=symbol_col, _as_of_col=None
                        )
            if last_exc is not None:
                raise last_exc
            raise
        raise


def calendar_days_to_trading_days(
    days: int, *, trading_days_per_year: int = 252
) -> int:
    """
    Convert calendar days to approximate trading days.

    Uses a simple linear scaling: trading ≈ round(days * 252 / 365).
    """

    if days <= 0:
        raise ValueError("days must be positive")
    return int(round(days * trading_days_per_year / 365.0))
