from __future__ import annotations

"""
End-to-end "Edge system" orchestration.

This module intentionally stays light-weight:
- Uses DB-API connections (sqlite / psycopg / psycopg2)
- Uses `layers.fetch_sp500_baseline` for the baseline universe
- Uses `features.compute_edge_features` for feature construction
- Adds a small "indicator application" layer to score baseline candidates in a
  regime-aware and horizon-aware way.
"""

from dataclasses import dataclass
from datetime import date, datetime
import re
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from .db import db_connection
from .features import EdgeFeatureConfig, compute_edge_features
from .layers import _as_iso_date, _validate_ident, fetch_sp500_baseline


_SYMBOL_RE = re.compile(r"^[A-Za-z0-9._\-]+$")


def _validate_symbols(symbols: Sequence[str]) -> list[str]:
    out: list[str] = []
    for s in symbols:
        sym = str(s or "").strip()
        if not sym:
            continue
        if not _SYMBOL_RE.fullmatch(sym):
            raise ValueError(f"Invalid symbol value: {s!r}")
        out.append(sym)
    # stable de-dupe
    seen: set[str] = set()
    deduped: list[str] = []
    for sym in out:
        if sym not in seen:
            seen.add(sym)
            deduped.append(sym)
    return deduped


def _conn_placeholder(conn: Any) -> str:
    """
    Best-effort DB-API placeholder style:
    - sqlite3: '?'
    - psycopg/psycopg2: '%s'
    """
    mod = getattr(conn.__class__, "__module__", "") or ""
    if "sqlite3" in mod:
        return "?"
    # psycopg (v3) and psycopg2 both accept %s
    return "%s"


def fetch_ohlcv_daily(
    conn: Any,
    *,
    symbols: Sequence[str],
    table: str = "ohlcv_daily",
    symbol_col: str = "symbol",
    date_col: str = "date",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV rows for a list of symbols.

    Expected table shape (defaults):
    - table: `ohlcv_daily`
    - columns: symbol, date, open, high, low, close, volume

    Returns a DataFrame with canonical column names:
      [symbol, date, open, high, low, close, volume]
    """
    t = _validate_ident(table, kind="table")
    symc = _validate_ident(symbol_col, kind="column")
    dc = _validate_ident(date_col, kind="column")
    oc = _validate_ident(open_col, kind="column")
    hc = _validate_ident(high_col, kind="column")
    lc = _validate_ident(low_col, kind="column")
    cc = _validate_ident(close_col, kind="column")
    vc = _validate_ident(volume_col, kind="column")

    syms = _validate_symbols(symbols)
    if not syms:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])

    ph = _conn_placeholder(conn)
    in_list = ", ".join([ph] * len(syms))
    where: list[str] = [f"{symc} IN ({in_list})"]
    params: list[Any] = list(syms)

    if start is not None:
        where.append(f"{dc} >= {ph}")
        params.append(_as_iso_date(start))
    if end is not None:
        where.append(f"{dc} <= {ph}")
        params.append(_as_iso_date(end))

    where_sql = " AND ".join(where)
    sql = f"""
    SELECT
      {symc} AS symbol,
      {dc} AS date,
      {oc} AS open,
      {hc} AS high,
      {lc} AS low,
      {cc} AS close,
      {vc} AS volume
    FROM {t}
    WHERE {where_sql}
    ORDER BY {symc}, {dc}
    """
    return pd.read_sql_query(sql, conn, params=params)


@dataclass(frozen=True)
class IndicatorSpec:
    """
    A cross-sectional indicator applied to candidate lists.

    - `source_col`: column in the features frame
    - `direction`: +1 for "higher is better", -1 for "lower is better"
    - `name`: optional friendly name (defaults to source_col)
    """

    source_col: str
    direction: float = 1.0
    name: str | None = None


@dataclass(frozen=True)
class EdgeEvaluation:
    """
    Controls how we build horizon-aware indicators for candidates.
    """

    horizon_days: tuple[int, ...] | None = None
    horizon_basis: Literal["trading", "calendar"] = "trading"
    regime_aware: bool = True
    indicators: tuple[IndicatorSpec, ...] = ()


def default_edge_evaluation(
    *,
    horizon_days: tuple[int, ...] | None = None,
    horizon_basis: Literal["trading", "calendar"] = "trading",
    regime_aware: bool = True,
) -> EdgeEvaluation:
    """
    A sensible default "evaluation system":
    - Structural: higher is better
    - Momentum: higher is better (horizon-aware)
    - Risk: lower vol/drawdown is better
    - Liquidity: higher is better
    """
    ind: list[IndicatorSpec] = [
        IndicatorSpec("struct_core", +1.0),
        IndicatorSpec("struct_growth_quality", +1.0),
        IndicatorSpec("struct_value_quality", +1.0),
        IndicatorSpec("ann_vol_20d", -1.0),
        IndicatorSpec("drawdown", +1.0),  # drawdown is negative; "less negative" is better
        IndicatorSpec("liquidity", +1.0),
    ]

    # momentum columns are created either from config.mom_lookbacks (mom_ret_21/63/126)
    # or from explicit horizon_days (mom_ret_{label}d). We add the horizon-aware specs later
    # when we see which columns exist.
    return EdgeEvaluation(
        horizon_days=horizon_days,
        horizon_basis=horizon_basis,
        regime_aware=regime_aware,
        indicators=tuple(ind),
    )


def _zscore(x: pd.Series) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce").astype("float64")
    mu = s.mean()
    sig = s.std(ddof=0)
    if not np.isfinite(sig) or sig == 0.0:
        return pd.Series(np.nan, index=s.index, dtype="float64")
    return (s - mu) / sig


def apply_indicators_to_candidates(
    candidates: pd.DataFrame,
    *,
    date_col: str = "date",
    regime_col: str = "regime",
    evaluation: EdgeEvaluation | None = None,
) -> pd.DataFrame:
    """
    Apply regime-aware + horizon-aware cross-sectional indicators to a candidate list.

    Output:
    - Adds indicator z-scores as columns named: `ind_z_{name}`
    - Adds a simple aggregate score: `edge_score` (mean of indicator z-scores)
    """
    if candidates.empty:
        out = candidates.copy()
        out["edge_score"] = pd.Series(dtype="float64")
        return out

    ev = evaluation or default_edge_evaluation()
    out = candidates.copy()

    # Horizon-aware: add momentum indicator specs for any momentum columns present.
    mom_cols = [c for c in out.columns if c.startswith("mom_ret_")]
    horizon_mom_specs = [IndicatorSpec(c, +1.0) for c in mom_cols]

    specs = list(ev.indicators) + horizon_mom_specs

    keys: list[str] = []
    if date_col in out.columns:
        keys.append(date_col)
    if ev.regime_aware and regime_col in out.columns:
        keys.append(regime_col)

    ind_cols: list[str] = []
    for spec in specs:
        col = spec.source_col
        if col not in out.columns:
            continue
        name = spec.name or col
        zname = f"ind_z_{name}"
        ind_cols.append(zname)

        if keys:
            z = (
                out.groupby(keys, dropna=False, sort=False)[col]
                .transform(_zscore)
                .astype("float64")
            )
        else:
            z = _zscore(out[col]).astype("float64")
        out[zname] = float(spec.direction) * z

    if ind_cols:
        out["edge_score"] = out[ind_cols].mean(axis=1)
    else:
        out["edge_score"] = np.nan

    return out


def build_candidate_list(
    features: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
    as_of: str | date | datetime | None = None,
) -> pd.DataFrame:
    """
    Reduce a per-symbol/per-date feature panel to a single "candidate row" per symbol.

    If `as_of` is provided, uses the last row per symbol with date <= as_of.
    Otherwise uses the last row per symbol.
    """
    if features.empty:
        return features.copy()

    df = features.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[symbol_col, date_col]).sort_values([symbol_col, date_col]).reset_index(drop=True)

    if as_of is not None:
        cutoff = pd.to_datetime(as_of, errors="raise")
        df = df[df[date_col] <= cutoff]

    if df.empty:
        return df

    idx = df.groupby(symbol_col, sort=False)[date_col].idxmax()
    return df.loc[idx].sort_values(symbol_col).reset_index(drop=True)


def run_edge(
    *,
    conn: Any | None = None,
    db_url: str | None = None,
    env_var: str = "SPRINGEDGE_DB_URL",
    # baseline (layers)
    baseline_table: str = "sp500",
    baseline_symbol_col: str = "symbol",
    baseline_as_of_col: str | None = "date",
    baseline_as_of: str | date | datetime | None = None,
    # ohlcv fetch
    ohlcv_table: str = "ohlcv_daily",
    ohlcv_symbol_col: str = "symbol",
    ohlcv_date_col: str = "date",
    ohlcv_start: str | date | datetime | None = None,
    ohlcv_end: str | date | datetime | None = None,
    # feature generation
    universe: str = "sp500",
    feature_config: EdgeFeatureConfig | None = None,
    horizon_days: tuple[int, ...] | None = None,
    horizon_basis: Literal["trading", "calendar"] = "trading",
    proxies: pd.DataFrame | None = None,
    structural: pd.DataFrame | None = None,
    # evaluation / indicator application
    evaluation: EdgeEvaluation | None = None,
    candidates_as_of: str | date | datetime | None = None,
) -> pd.DataFrame:
    """
    End-to-end "connect everything":
    - Connect DB (unless `conn` provided)
    - Pull baseline symbols from layers
    - Fetch OHLCV for baseline symbols
    - Generate features dynamically (optionally horizon-aware via horizon_days/basis)
    - Build a candidate list (one row per symbol)
    - Apply regime-aware and horizon-aware indicators on the candidates
    """

    def _run(c: Any) -> pd.DataFrame:
        baseline = fetch_sp500_baseline(
            c,
            table=baseline_table,
            symbol_col=baseline_symbol_col,
            as_of_col=baseline_as_of_col,
            as_of=baseline_as_of,
        )
        symbols = baseline[baseline_symbol_col].astype("string").dropna().tolist()
        ohlcv = fetch_ohlcv_daily(
            c,
            symbols=symbols,
            table=ohlcv_table,
            symbol_col=ohlcv_symbol_col,
            date_col=ohlcv_date_col,
            start=ohlcv_start,
            end=ohlcv_end,
        )
        feats = compute_edge_features(
            ohlcv,
            universe=universe,
            config=feature_config,
            horizon_days=horizon_days,
            horizon_basis=horizon_basis,
            proxies=proxies,
            structural=structural,
        )
        candidates = build_candidate_list(feats, symbol_col="symbol", date_col="date", as_of=candidates_as_of)
        return apply_indicators_to_candidates(candidates, evaluation=evaluation)

    if conn is not None:
        return _run(conn)
    with db_connection(db_url, env_var=env_var) as c:
        return _run(c)

