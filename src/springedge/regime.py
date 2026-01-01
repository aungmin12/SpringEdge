from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd

from .layers import _as_iso_date, _validate_ident, _validate_table_ref


def fetch_regime_daily(
    conn: Any,
    *,
    table: str = "regime_daily",
    date_col: str = "date",
    regime_col: str = "regime",
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
) -> pd.DataFrame:
    """
    Fetch daily regime labels from the DB.

    Expected table shape (defaults):
    - table: `regime_daily`
    - date column: `date`
    - regime label column: `regime`

    Returns a DataFrame with columns: [date_col, regime_col], ordered by date.
    """
    t = _validate_table_ref(table, kind="table")
    d = _validate_ident(date_col, kind="column")
    r = _validate_ident(regime_col, kind="column")

    where: list[str] = []
    if start is not None:
        where.append(f"{d} >= '{_as_iso_date(start)}'")
    if end is not None:
        where.append(f"{d} <= '{_as_iso_date(end)}'")

    where_sql = f" WHERE {' AND '.join(where)}" if where else ""
    sql = f"SELECT {d} AS {d}, {r} AS {r} FROM {t}{where_sql} ORDER BY {d}"
    cur = conn.cursor()
    try:
        try:
            cur.execute(sql)
        except Exception:
            try:
                if hasattr(conn, "rollback"):
                    conn.rollback()
            except Exception:
                pass
            raise
        rows = cur.fetchall()
        cols = [desc[0] for desc in (cur.description or [])]
    finally:
        try:
            cur.close()
        except Exception:
            pass
    return pd.DataFrame.from_records(rows, columns=cols)


@dataclass(frozen=True)
class QuarterDominantRegime:
    quarter: str
    dominant_regime: str
    dominant_days: int
    total_days: int


def quarterly_regime_profile(
    regime_daily: pd.DataFrame,
    *,
    date_col: str = "date",
    regime_col: str = "regime",
    quarter: str | pd.Period | None = None,
) -> tuple[pd.DataFrame, QuarterDominantRegime]:
    """
    Build a quarterly regime profile from daily regime labels.

    Output:
    - profile: long-form DataFrame for the selected quarter with columns:
        [quarter, regime, n_days, is_dominant]
    - dominant: a small summary object for the selected (default: most recent) quarter
    """
    if date_col not in regime_daily.columns or regime_col not in regime_daily.columns:
        raise ValueError(f"regime_daily must include columns {date_col!r} and {regime_col!r}")

    df = regime_daily[[date_col, regime_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, regime_col])
    df[regime_col] = df[regime_col].astype("string")
    df = df.sort_values(date_col).reset_index(drop=True)

    if df.empty:
        raise ValueError("regime_daily is empty after coercion; cannot compute quarterly profile")

    qser = df[date_col].dt.to_period("Q")
    df["_quarter"] = qser

    if quarter is None:
        q = df["_quarter"].max()
    elif isinstance(quarter, pd.Period):
        q = quarter.asfreq("Q")
    else:
        s = str(quarter).strip()
        if "Q" in s.upper():
            q = pd.Period(s, freq="Q")
        else:
            q = pd.to_datetime(s, errors="raise").to_period("Q")

    qdf = df[df["_quarter"] == q].copy()
    if qdf.empty:
        raise ValueError(f"No regime_daily rows found for quarter={str(q)!r}")

    counts = (
        qdf.groupby(regime_col, dropna=True)
        .size()
        .rename("n_days")
        .reset_index()
        .sort_values(["n_days", regime_col], ascending=[False, True])
        .reset_index(drop=True)
    )

    dominant_regime = str(counts.loc[0, regime_col])
    dominant_days = int(counts.loc[0, "n_days"])
    total_days = int(counts["n_days"].sum())
    counts["quarter"] = str(q)
    counts["is_dominant"] = counts[regime_col].astype("string") == dominant_regime
    profile = counts[["quarter", regime_col, "n_days", "is_dominant"]].rename(
        columns={regime_col: "regime"}
    )

    return (
        profile,
        QuarterDominantRegime(
            quarter=str(q),
            dominant_regime=dominant_regime,
            dominant_days=dominant_days,
            total_days=total_days,
        ),
    )

