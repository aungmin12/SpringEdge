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

import os
import sys
from dataclasses import dataclass
from datetime import date, datetime
import argparse
import logging
import re
from typing import Any, Iterable, Literal, Sequence

# Allow running this file directly (e.g. `python3 src/springedge/edge.py`).
# Recommended invocation remains: `python3 -m springedge.edge`.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    _pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
    __package__ = "springedge"

import numpy as np
import pandas as pd

from .db import db_connection
from .features import EdgeFeatureConfig, compute_edge_features
from .layers import _as_iso_date, _validate_ident, fetch_sp500_baseline


_SYMBOL_RE = re.compile(r"^[A-Za-z0-9._\-]+$")
_LOG = logging.getLogger(__name__)


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
        _LOG.info(
            "run_edge: baseline_table=%s baseline_as_of=%s ohlcv_table=%s ohlcv_start=%s ohlcv_end=%s horizon_days=%s horizon_basis=%s",
            baseline_table,
            baseline_as_of,
            ohlcv_table,
            ohlcv_start,
            ohlcv_end,
            horizon_days,
            horizon_basis,
        )
        baseline = fetch_sp500_baseline(
            c,
            table=baseline_table,
            symbol_col=baseline_symbol_col,
            as_of_col=baseline_as_of_col,
            as_of=baseline_as_of,
        )
        symbols = baseline[baseline_symbol_col].astype("string").dropna().tolist()
        _LOG.info("run_edge: baseline symbols=%d", len(symbols))
        ohlcv = fetch_ohlcv_daily(
            c,
            symbols=symbols,
            table=ohlcv_table,
            symbol_col=ohlcv_symbol_col,
            date_col=ohlcv_date_col,
            start=ohlcv_start,
            end=ohlcv_end,
        )
        _LOG.info("run_edge: ohlcv rows=%d", len(ohlcv))
        feats = compute_edge_features(
            ohlcv,
            universe=universe,
            config=feature_config,
            horizon_days=horizon_days,
            horizon_basis=horizon_basis,
            proxies=proxies,
            structural=structural,
        )
        _LOG.info("run_edge: feature rows=%d cols=%d", len(feats), feats.shape[1] if not feats.empty else 0)
        candidates = build_candidate_list(feats, symbol_col="symbol", date_col="date", as_of=candidates_as_of)
        _LOG.info("run_edge: candidates=%d", len(candidates))
        scored = apply_indicators_to_candidates(candidates, evaluation=evaluation)
        _LOG.info("run_edge: scored candidates=%d (edge_score present=%s)", len(scored), "edge_score" in scored.columns)
        return scored

    if conn is not None:
        return _run(conn)
    with db_connection(db_url, env_var=env_var) as c:
        return _run(c)


def _parse_horizon_days(values: Sequence[str] | None) -> tuple[int, ...] | None:
    """
    Accept either repeated ints (e.g. `--horizon-days 7 21`) or a single comma string (`--horizon-days 7,21`).
    """
    if not values:
        return None
    raw = ",".join([v.strip() for v in values if str(v).strip() != ""])
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        out.append(int(p))
    return tuple(out)


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI entrypoint.

    Examples:
      - Demo (no DB required):
        python3 -m springedge.edge --demo

      - Real DB:
        export SPRINGEDGE_DB_URL="postgresql://..."
        python3 -m springedge.edge --baseline-as-of 2024-02-01 --horizon-days 7 21
    """
    p = argparse.ArgumentParser(prog="springedge.edge", description="Run end-to-end Edge orchestration and print scored candidates.")
    p.add_argument("--log-level", default="INFO", help="Logging level (e.g. DEBUG, INFO, WARNING). Default: INFO.")
    p.add_argument("--demo", action="store_true", help="Run a self-contained sqlite demo (no external DB required).")

    # DB / baseline / ohlcv
    p.add_argument("--db-url", default=None, help="Database URL (overrides env var if provided).")
    p.add_argument("--env-var", default="SPRINGEDGE_DB_URL", help="Env var containing DB URL. Default: SPRINGEDGE_DB_URL.")
    p.add_argument("--baseline-table", default="sp500")
    p.add_argument("--baseline-symbol-col", default="symbol")
    p.add_argument("--baseline-as-of-col", default="date")
    p.add_argument("--baseline-as-of", default=None, help="Baseline snapshot as-of date (e.g. 2024-02-01).")
    p.add_argument("--ohlcv-table", default="ohlcv_daily")
    p.add_argument("--ohlcv-symbol-col", default="symbol")
    p.add_argument("--ohlcv-date-col", default="date")
    p.add_argument("--ohlcv-start", default=None, help="OHLCV start date filter (inclusive).")
    p.add_argument("--ohlcv-end", default=None, help="OHLCV end date filter (inclusive).")

    # Features / evaluation
    p.add_argument("--universe", default="sp500")
    p.add_argument(
        "--horizon-days",
        nargs="*",
        default=None,
        help="Horizon days (e.g. `--horizon-days 7 21` or `--horizon-days 7,21`).",
    )
    p.add_argument("--horizon-basis", choices=("trading", "calendar"), default="trading")
    p.add_argument("--candidates-as-of", default=None, help="Build one row per symbol as-of this date.")
    p.add_argument("--top", type=int, default=25, help="How many candidates to print. Default: 25.")
    args = p.parse_args(list(argv) if argv is not None else None)

    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    horizon_days = _parse_horizon_days(args.horizon_days)

    if args.demo:
        import sqlite3

        _LOG.info("Running demo sqlite pipeline.")
        conn = sqlite3.connect(":memory:")
        conn.execute("create table sp500 (date text, symbol text)")
        conn.execute(
            "create table ohlcv_daily (symbol text, date text, open real, high real, low real, close real, volume real)"
        )
        conn.executemany(
            "insert into sp500 (date, symbol) values (?, ?)",
            [
                ("2024-01-02", "AAA"),
                ("2024-01-02", "BBB"),
                ("2024-02-01", "AAA"),
                ("2024-02-01", "BBB"),
            ],
        )
        dates = pd.bdate_range("2023-01-02", periods=320)
        for sym, base in [("AAA", 100.0), ("BBB", 50.0)]:
            close = base * np.exp(np.cumsum(np.full(len(dates), 0.001)))
            open_ = np.r_[close[0], close[:-1]]
            high = np.maximum(open_, close) * 1.01
            low = np.minimum(open_, close) * 0.99
            vol = np.full(len(dates), 1_000_000.0)
            rows = list(
                zip(
                    [sym] * len(dates),
                    [d.date().isoformat() for d in dates],
                    open_.astype(float),
                    high.astype(float),
                    low.astype(float),
                    close.astype(float),
                    vol.astype(float),
                )
            )
            conn.executemany(
                "insert into ohlcv_daily (symbol, date, open, high, low, close, volume) values (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        out = run_edge(
            conn=conn,
            baseline_table=args.baseline_table,
            baseline_symbol_col=args.baseline_symbol_col,
            baseline_as_of_col=args.baseline_as_of_col,
            baseline_as_of=args.baseline_as_of or "2024-02-01",
            ohlcv_table=args.ohlcv_table,
            ohlcv_symbol_col=args.ohlcv_symbol_col,
            ohlcv_date_col=args.ohlcv_date_col,
            ohlcv_start=args.ohlcv_start,
            ohlcv_end=args.ohlcv_end,
            universe=args.universe,
            horizon_days=horizon_days or (7, 21),
            horizon_basis=args.horizon_basis,
            candidates_as_of=args.candidates_as_of,
        )
    else:
        out = run_edge(
            db_url=args.db_url,
            env_var=args.env_var,
            baseline_table=args.baseline_table,
            baseline_symbol_col=args.baseline_symbol_col,
            baseline_as_of_col=args.baseline_as_of_col,
            baseline_as_of=args.baseline_as_of,
            ohlcv_table=args.ohlcv_table,
            ohlcv_symbol_col=args.ohlcv_symbol_col,
            ohlcv_date_col=args.ohlcv_date_col,
            ohlcv_start=args.ohlcv_start,
            ohlcv_end=args.ohlcv_end,
            universe=args.universe,
            horizon_days=horizon_days,
            horizon_basis=args.horizon_basis,
            candidates_as_of=args.candidates_as_of,
        )

    if out is None or out.empty:
        _LOG.warning("No candidates returned.")
        return 2

    if "edge_score" in out.columns:
        view = out.sort_values("edge_score", ascending=False, kind="mergesort").reset_index(drop=True)
    else:
        view = out.reset_index(drop=True)

    cols = [c for c in ["symbol", "date", "regime", "edge_score"] if c in view.columns]
    # Add momentum columns if present (nice quick signal).
    cols += [c for c in view.columns if c.startswith("mom_ret_")]
    cols = list(dict.fromkeys(cols))  # stable de-dupe
    print(view.loc[: max(args.top - 1, 0), cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
