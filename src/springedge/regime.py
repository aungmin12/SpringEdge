from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
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


def fetch_market_regime_daily(
    conn: Any,
    *,
    table: str = "market_regime_daily",
    analysis_date_col: str = "analysis_date",
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
) -> pd.DataFrame:
    """
    Fetch daily market regime data from the DB.

    Expected table shape (as provided by the user):
    - `analysis_date`
    - `regime_label`
    - `regime_score`
    - `position_multiplier`
    - `vix_level`
    - `vix_zscore`
    - `vix9d_over_vix`
    - `vix_vix3m_ratio`
    - `move_level`
    - `move_zscore`
    - `term_structure_state`
    - `event_risk_flag`
    - `joint_stress_flag`
    - `notes`
    - `created_at`

    Returns a DataFrame ordered by analysis_date ascending.
    """

    def _safe_rollback() -> None:
        try:
            if hasattr(conn, "rollback"):
                conn.rollback()
        except Exception:
            pass

    def _missing_table_error(err: Exception, *, table_name: str) -> bool:
        msg = str(err).lower()
        tn = str(table_name).lower()
        candidates = {tn}
        if "." in tn:
            candidates.add(tn.split(".")[-1])
        return any((f'relation "{c}" does not exist' in msg) or (f"no such table: {c}" in msg) for c in candidates)

    def _run_query(*, _table: str) -> pd.DataFrame:
        t = _validate_table_ref(_table, kind="table")
        d = _validate_ident(analysis_date_col, kind="column")
        cols = [
            d,
            _validate_ident("regime_label", kind="column"),
            _validate_ident("regime_score", kind="column"),
            _validate_ident("position_multiplier", kind="column"),
            _validate_ident("vix_level", kind="column"),
            _validate_ident("vix_zscore", kind="column"),
            _validate_ident("vix9d_over_vix", kind="column"),
            _validate_ident("vix_vix3m_ratio", kind="column"),
            _validate_ident("move_level", kind="column"),
            _validate_ident("move_zscore", kind="column"),
            _validate_ident("term_structure_state", kind="column"),
            _validate_ident("event_risk_flag", kind="column"),
            _validate_ident("joint_stress_flag", kind="column"),
            _validate_ident("notes", kind="column"),
            _validate_ident("created_at", kind="column"),
        ]

        where: list[str] = []
        if start is not None:
            where.append(f"{d} >= '{_as_iso_date(start)}'")
        if end is not None:
            where.append(f"{d} <= '{_as_iso_date(end)}'")
        where_sql = f" WHERE {' AND '.join(where)}" if where else ""
        sql = f"SELECT {', '.join(cols)} FROM {t}{where_sql} ORDER BY {d}"

        cur = conn.cursor()
        try:
            try:
                cur.execute(sql)
            except Exception:
                _safe_rollback()
                raise
            rows = cur.fetchall()
            out_cols = [desc[0] for desc in (cur.description or [])]
        finally:
            try:
                cur.close()
            except Exception:
                pass
        return pd.DataFrame.from_records(rows, columns=out_cols)

    try:
        return _run_query(_table=table)
    except Exception as exc:
        _safe_rollback()
        # Common production convention: schema-qualified intelligence table.
        candidates = (table, "intelligence.market_regime_daily")
        last_exc: Exception | None = None
        for cand in candidates:
            if cand == table:
                continue
            try:
                return _run_query(_table=cand)
            except Exception as exc2:
                last_exc = exc2
                _safe_rollback()
                if _missing_table_error(exc2, table_name=cand):
                    continue
                break
        if last_exc is not None and _missing_table_error(exc, table_name=table):
            raise last_exc
        raise


@dataclass(frozen=True)
class MarketRegimeSummary:
    as_of: str
    lookback_days: int
    dominant_regime: str
    dominant_days: int
    total_days: int
    latest_regime: str
    latest_streak_days: int
    recent_share_latest: float
    recent_score_slope: float | None


def summarize_market_regime(
    market_regime_daily: pd.DataFrame,
    *,
    as_of: str | date | datetime | None = None,
    lookback_days: int = 90,
    trend_lookback_days: int = 20,
    date_col: str = "analysis_date",
    regime_col: str = "regime_label",
    score_col: str = "regime_score",
) -> tuple[pd.DataFrame, MarketRegimeSummary]:
    """
    Compute:
    - Dominant market regime over the past `lookback_days` (calendar days)
    - Latest regime and a simple "trend" read:
        - consecutive streak length of the latest regime
        - share of last `trend_lookback_days` that match latest regime
        - slope of `regime_score` over last `trend_lookback_days` (best-effort)

    Returns:
    - counts: DataFrame with columns [regime, n_days, share]
    - summary: MarketRegimeSummary
    """
    if date_col not in market_regime_daily.columns or regime_col not in market_regime_daily.columns:
        raise ValueError(f"market_regime_daily must include columns {date_col!r} and {regime_col!r}")

    df = market_regime_daily.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, regime_col]).sort_values(date_col).reset_index(drop=True)
    df[regime_col] = df[regime_col].astype("string")
    if score_col in df.columns:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce").astype("float64")

    if df.empty:
        raise ValueError("market_regime_daily is empty after coercion; cannot compute summary")

    as_of_ts = pd.to_datetime(as_of, errors="raise") if as_of is not None else df[date_col].max()
    start_ts = as_of_ts - timedelta(days=int(lookback_days))
    w = df[(df[date_col] >= start_ts) & (df[date_col] <= as_of_ts)].copy()
    if w.empty:
        raise ValueError("No market_regime_daily rows found in the requested lookback window")

    counts = (
        w.groupby(regime_col, dropna=True)
        .size()
        .rename("n_days")
        .reset_index()
        .sort_values(["n_days", regime_col], ascending=[False, True])
        .reset_index(drop=True)
    )
    total_days = int(counts["n_days"].sum())
    counts["share"] = counts["n_days"] / float(total_days) if total_days else 0.0
    counts = counts.rename(columns={regime_col: "regime"})

    dominant_regime = str(counts.loc[0, "regime"])
    dominant_days = int(counts.loc[0, "n_days"])

    # Latest regime and trend features
    latest_row = w.loc[w[date_col].idxmax()]
    latest_regime = str(latest_row[regime_col])

    # Latest streak: count consecutive days from the end with the same regime_label.
    rev = w.sort_values(date_col, ascending=False).reset_index(drop=True)
    streak = 0
    for _, row in rev.iterrows():
        if str(row[regime_col]) == latest_regime:
            streak += 1
        else:
            break

    recent = rev.head(int(trend_lookback_days)).copy()
    recent_n = int(len(recent))
    recent_share_latest = float((recent[regime_col].astype("string") == latest_regime).mean()) if recent_n else 0.0

    # Score slope over recent window (simple least squares on index).
    recent_score_slope: float | None = None
    if score_col in recent.columns:
        s = pd.to_numeric(recent[score_col], errors="coerce").astype("float64")
        s = s.reset_index(drop=True)
        ok = s.notna()
        if int(ok.sum()) >= 2:
            x = pd.Series(range(len(s)), dtype="float64")
            x = x[ok]
            y = s[ok]
            # slope = cov(x,y)/var(x)
            denom = float(((x - x.mean()) ** 2).sum())
            if denom != 0.0:
                recent_score_slope = float(((x - x.mean()) * (y - y.mean())).sum() / denom)

    summary = MarketRegimeSummary(
        as_of=pd.to_datetime(as_of_ts).date().isoformat(),
        lookback_days=int(lookback_days),
        dominant_regime=dominant_regime,
        dominant_days=dominant_days,
        total_days=total_days,
        latest_regime=latest_regime,
        latest_streak_days=int(streak),
        recent_share_latest=float(recent_share_latest),
        recent_score_slope=recent_score_slope,
    )
    return counts, summary


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

