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

# If this file is executed directly (e.g. `python3 src/springedge/edge.py` or
# running it from within `src/springedge/`), Python does not treat it as part of
# the `springedge` package, so relative imports like `from .db import ...` fail
# with: "attempted relative import with no known parent package".
#
# We fix that by (1) ensuring `src/` is on sys.path and (2) setting __package__
# so relative imports resolve correctly. Normal package imports (`python -m
# springedge.edge` or `import springedge`) are unaffected.
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    import os
    import sys

    _src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _src not in sys.path:
        sys.path.insert(0, _src)
    __package__ = "springedge"

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from .db import db_connection
from .logging_utils import configure_logging
from .features import EdgeFeatureConfig, compute_edge_features
from .layers import (
    _as_iso_date,
    _validate_ident,
    _validate_table_ref,
    fetch_sp500_baseline,
)
from .regime import fetch_market_regime_daily, summarize_market_regime


_SYMBOL_RE = re.compile(r"^[A-Za-z0-9._\-]+$")
# If a user runs this file directly (`python3 src/springedge/edge.py`), `__name__`
# becomes "__main__", which looks bad in logs. Use a stable logger name instead.
_LOG = logging.getLogger("springedge.edge" if __name__ == "__main__" else __name__)
_LOG_SYMBOLS_PREVIEW_LIMIT = 50


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
    table: str = "core.prices_daily",
    symbol_col: str = "symbol",
    date_col: str = "date",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
    # Optional compatibility: prices table keyed by security_id + trade_date.
    # If your `prices_daily` table has `security_id` instead of `symbol`, we can
    # join through `core.security` (or `security`) to map ticker symbols to IDs.
    security_table: str = "core.security",
    security_id_col: str = "security_id",
    security_symbol_col: str = "symbol",
    prices_security_id_col: str = "security_id",
    prices_trade_date_col: str = "trade_date",
) -> pd.DataFrame:
    """
    Fetch daily OHLCV rows for a list of symbols.

    Expected table shape (defaults):
    - table: `core.prices_daily`
    - columns: symbol, date, open, high, low, close, volume

    Supported production variant (common in normalized schemas):
    - prices table columns: security_id, trade_date, open, high, low, close, volume, ...
    - security table columns: security_id, symbol
    In that case, we join prices->security so callers can still pass ticker symbols.

    Returns a DataFrame with canonical column names:
      [symbol, date, open, high, low, close, volume]
    """

    def _safe_rollback() -> None:
        """
        Best-effort rollback for drivers that abort the current transaction on errors.

        Note: sqlite will rollback *all uncommitted fixture inserts* if we call rollback()
        after a SELECT error (e.g. missing column during fallback attempts). For sqlite
        connections, we skip rollback.
        """
        try:
            mod = getattr(conn.__class__, "__module__", "") or ""
            if "sqlite3" in mod:
                return
            if hasattr(conn, "rollback"):
                conn.rollback()
        except Exception:
            pass

    def _fetch_df(sql: str, params: Sequence[Any]) -> pd.DataFrame:
        """
        Execute a parametrized SQL query via a DB-API cursor and return a DataFrame.

        Avoids `pandas.read_sql_query` warnings for some DB-API connections
        (e.g. psycopg v3) unless SQLAlchemy is used.
        """
        cur = conn.cursor()
        try:
            try:
                cur.execute(sql, list(params))
            except Exception:
                # psycopg aborts transactions on errors; rollback allows safe retry/fallback.
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

    t = _validate_table_ref(table, kind="table")
    symc = _validate_ident(symbol_col, kind="column")
    dc = _validate_ident(date_col, kind="column")
    oc = _validate_ident(open_col, kind="column")
    hc = _validate_ident(high_col, kind="column")
    lc = _validate_ident(low_col, kind="column")
    cc = _validate_ident(close_col, kind="column")
    vc = _validate_ident(volume_col, kind="column")

    syms = _validate_symbols(symbols)
    if not syms:
        return pd.DataFrame(
            columns=["symbol", "date", "open", "high", "low", "close", "volume"]
        )

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
    try:
        return _fetch_df(sql, params)
    except Exception as exc:
        # NOTE: exception variables are cleared at the end of an `except` block.
        # Capture it to a normal local so nested helpers can safely reference it.
        exc_value: Exception = exc

        # Compatibility: some schemas use different column names for ticker/date.
        # Common example: `ticker` instead of `symbol` in a `prices_daily` table.
        #
        # We only attempt these aliases when the caller is using the defaults
        # (`symbol`/`date`). If the caller explicitly overrides, we assume they
        # know their schema and avoid endless retry loops.
        def _try_fetch_with_column_aliases() -> pd.DataFrame | None:
            sym_missing = _missing_column_error(exc_value, column_name=symbol_col)
            date_missing = _missing_column_error(exc_value, column_name=date_col)
            if not (sym_missing or date_missing):
                return None

            sym_aliases: tuple[str, ...] = ()
            if sym_missing and symbol_col == "symbol":
                sym_aliases = ("ticker", "sym", "tsym", "ticker_symbol")

            date_aliases: tuple[str, ...] = ()
            if date_missing and date_col == "date":
                date_aliases = ("trade_date", "dt", "asof_date")

            if not sym_aliases and not date_aliases:
                return None

            # Try combinations, but keep the original values first.
            sym_candidates = (symbol_col,) + sym_aliases
            date_candidates = (date_col,) + date_aliases

            last_exc: Exception | None = None
            for symc2 in sym_candidates:
                for dc2 in date_candidates:
                    # Skip the exact same pair.
                    if symc2 == symbol_col and dc2 == date_col:
                        continue
                    try:
                        return fetch_ohlcv_daily(
                            conn,
                            symbols=symbols,
                            table=table,
                            symbol_col=symc2,
                            date_col=dc2,
                            open_col=open_col,
                            high_col=high_col,
                            low_col=low_col,
                            close_col=close_col,
                            volume_col=volume_col,
                            start=start,
                            end=end,
                            security_table=security_table,
                            security_id_col=security_id_col,
                            security_symbol_col=security_symbol_col,
                            prices_security_id_col=prices_security_id_col,
                            prices_trade_date_col=prices_trade_date_col,
                        )
                    except Exception as exc2:
                        last_exc = exc2
                        # If we still have missing column errors, keep trying.
                        if symc2 != symbol_col and _missing_column_error(
                            exc2, column_name=symc2
                        ):
                            continue
                        if dc2 != date_col and _missing_column_error(
                            exc2, column_name=dc2
                        ):
                            continue
                        # For other errors (or mismatches), stop early.
                        break
                else:
                    continue
                break

            if last_exc is not None:
                # Don't hide unexpected errors; return None so outer handler can proceed.
                return None
            return None

        # Compatibility: if the target table doesn't have the assumed column names,
        # retry a common production schema:
        # - prices table: security_id + trade_date
        # - join through security table to filter by ticker symbols
        def _try_fetch_joined_security() -> pd.DataFrame | None:
            # Only attempt this when the caller is filtering by ticker symbols.
            if symbol_col != "symbol":
                return None
            if not (
                _missing_column_error(exc_value, column_name=symbol_col)
                or _missing_column_error(exc_value, column_name=date_col)
            ):
                return None

            pt = _validate_table_ref(table, kind="table")
            psid = _validate_ident(prices_security_id_col, kind="column")
            pdc = _validate_ident(prices_trade_date_col, kind="column")
            sid = _validate_ident(security_id_col, kind="column")
            ssym = _validate_ident(security_symbol_col, kind="column")

            # Prefer the configured security_table, but fall back to a common alias.
            sec_tables = (security_table, "security", "core.security")
            last_exc: Exception | None = None

            join_where: list[str] = [f"s.{ssym} IN ({in_list})"]
            join_params: list[Any] = list(syms)
            if start is not None:
                join_where.append(f"p.{pdc} >= {ph}")
                join_params.append(_as_iso_date(start))
            if end is not None:
                join_where.append(f"p.{pdc} <= {ph}")
                join_params.append(_as_iso_date(end))
            join_where_sql = " AND ".join(join_where)

            for st in sec_tables:
                try:
                    stq = _validate_table_ref(st, kind="table")
                except Exception:
                    continue
                join_sql = f"""
                SELECT
                  s.{ssym} AS symbol,
                  p.{pdc} AS date,
                  p.{oc} AS open,
                  p.{hc} AS high,
                  p.{lc} AS low,
                  p.{cc} AS close,
                  p.{vc} AS volume
                FROM {pt} p
                JOIN {stq} s
                  ON p.{psid} = s.{sid}
                WHERE {join_where_sql}
                ORDER BY s.{ssym}, p.{pdc}
                """
                try:
                    return _fetch_df(join_sql, join_params)
                except Exception as exc2:
                    last_exc = exc2
                    _safe_rollback()
                    # Keep trying other likely security table names if the table is missing.
                    if _missing_table_error(exc2, table_name=st):
                        continue
                    # If the join/columns don't exist, don't swallow it silently.
                    # We'll fall back to the original error below.
                    break

            if last_exc is not None:
                raise last_exc
            return None

        # psycopg aborts transactions on errors (even for SELECT). If we are going
        # to retry a fallback table, we must rollback first or subsequent queries
        # can fail with InFailedSqlTransaction.
        _safe_rollback()

        # Next, try common "ticker/date" alias columns on the same prices table.
        aliased = _try_fetch_with_column_aliases()
        if aliased is not None:
            return aliased

        # First, try the joined-security compatibility path.
        joined = _try_fetch_joined_security()
        if joined is not None:
            return joined

        # Production schema compatibility: many DBs store daily prices in a schema,
        # e.g. `core.prices_daily`, while others expose it as `prices_daily` or `ohlcv_daily`.
        if table == "core.prices_daily" and _missing_table_error(
            exc, table_name="core.prices_daily"
        ):
            for cand in ("prices_daily", "ohlcv_daily"):
                try:
                    return fetch_ohlcv_daily(
                        conn,
                        symbols=symbols,
                        table=cand,
                        symbol_col=symbol_col,
                        date_col=date_col,
                        open_col=open_col,
                        high_col=high_col,
                        low_col=low_col,
                        close_col=close_col,
                        volume_col=volume_col,
                        start=start,
                        end=end,
                        security_table=security_table,
                        security_id_col=security_id_col,
                        security_symbol_col=security_symbol_col,
                        prices_security_id_col=prices_security_id_col,
                        prices_trade_date_col=prices_trade_date_col,
                    )
                except Exception as exc2:
                    if _missing_table_error(exc2, table_name=cand):
                        continue
                    raise
        # Production schema compatibility: some DBs name this table `prices_daily`,
        # and some schema-qualify it under `core`.
        if table == "ohlcv_daily" and _missing_table_error(
            exc, table_name="ohlcv_daily"
        ):
            for cand in ("prices_daily", "core.prices_daily"):
                try:
                    return fetch_ohlcv_daily(
                        conn,
                        symbols=symbols,
                        table=cand,
                        symbol_col=symbol_col,
                        date_col=date_col,
                        open_col=open_col,
                        high_col=high_col,
                        low_col=low_col,
                        close_col=close_col,
                        volume_col=volume_col,
                        start=start,
                        end=end,
                        security_table=security_table,
                        security_id_col=security_id_col,
                        security_symbol_col=security_symbol_col,
                        prices_security_id_col=prices_security_id_col,
                        prices_trade_date_col=prices_trade_date_col,
                    )
                except Exception as exc2:
                    if _missing_table_error(exc2, table_name=cand):
                        continue
                    raise
        if table == "prices_daily" and _missing_table_error(
            exc, table_name="prices_daily"
        ):
            # If user explicitly set table=prices_daily, still try schema-qualified.
            return fetch_ohlcv_daily(
                conn,
                symbols=symbols,
                table="core.prices_daily",
                symbol_col=symbol_col,
                date_col=date_col,
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
                volume_col=volume_col,
                start=start,
                end=end,
                security_table=security_table,
                security_id_col=security_id_col,
                security_symbol_col=security_symbol_col,
                prices_security_id_col=prices_security_id_col,
                prices_trade_date_col=prices_trade_date_col,
            )
        raise


def _missing_table_error(err: Exception, *, table_name: str) -> bool:
    """
    Best-effort detection of missing table errors across DB drivers.
    - Postgres (psycopg/psycopg2): relation "x" does not exist
    - sqlite: no such table: x
    """
    msg = str(err).lower()
    tn = str(table_name).lower()
    return (f'relation "{tn}" does not exist' in msg) or (f"no such table: {tn}" in msg)


def fetch_symbol_universe_from_table(
    conn: Any,
    *,
    table: str,
    symbol_col: str = "symbol",
    out_col: str = "symbol",
) -> pd.DataFrame:
    """
    Fallback universe loader: infer a baseline universe as DISTINCT symbols from a table.

    This is useful when a baseline membership table (like `sp500`) does not exist
    but you *do* have a prices/OHLCV table.
    """
    t = _validate_table_ref(table, kind="table")
    symc = _validate_ident(symbol_col, kind="column")
    outc = _validate_ident(out_col, kind="column")
    sql = f"SELECT DISTINCT {symc} AS {outc} FROM {t} ORDER BY {symc}"
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
        cols = [d[0] for d in (cur.description or [])]
    finally:
        try:
            cur.close()
        except Exception:
            pass
    df = pd.DataFrame.from_records(rows, columns=cols)
    if outc not in df.columns:
        df.columns = [str(c) for c in df.columns]
        if outc not in df.columns and len(df.columns) == 1:
            df = df.rename(columns={df.columns[0]: outc})
    df[outc] = df[outc].astype("string")
    return df.dropna().drop_duplicates().sort_values(outc).reset_index(drop=True)


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


@dataclass(frozen=True)
class TopDownRegimeRule:
    """
    Regime-specific configuration for top-down horizon integration.

    Intended interpretation (your roles):
    - 365D: Business quality
    - 63–126D: Structural momentum
    - 30D: Trend-continuation confirmation
    - 21D: Tactical trigger
    - 7D: Micro timing / noise
    """

    # Relative weights applied to layer z-scores.
    weight_quality_365: float = 1.0
    weight_structural_63_126: float = 1.0
    weight_confirm_30: float = 1.0
    weight_trigger_21: float = 1.0
    weight_micro_7: float = 0.5

    # How to interpret 7D micro timing:
    # - "follow": positive micro momentum helps (trend-following entry)
    # - "pullback": negative micro momentum helps (buy weakness in uptrend)
    # - "ignore": do not use micro layer
    micro_mode: Literal["follow", "pullback", "ignore"] = "pullback"

    # Soft gating strength. Higher => long-horizon layers gate short-horizon layers more.
    gate_strength: float = 1.0


@dataclass(frozen=True)
class TopDownEvaluation:
    """
    A regime-aware, top-down (365→…→7) evaluation model.

    Pass this to `run_edge(..., evaluation=TopDownEvaluation(...))`.
    """

    regime_aware: bool = True

    # 365D business quality proxy columns (structural/fundamental composites).
    quality_cols: tuple[str, ...] = (
        "struct_core",
        "struct_growth_quality",
        "struct_value_quality",
    )

    # Horizon roles expressed in *day labels* that correspond to feature columns:
    # - `mom_ret_{N}d` when `horizon_days` is used
    # - or `mom_ret_{N}` for the built-in defaults (21/63/126)
    micro_days: tuple[int, ...] = (7,)
    trigger_days: tuple[int, ...] = (21,)
    confirm_days: tuple[int, ...] = (30,)
    structural_days: tuple[int, ...] = (63, 126)

    # Default rule + optional overrides by regime label.
    default_rule: TopDownRegimeRule = field(default_factory=TopDownRegimeRule)
    regime_rules: dict[str, TopDownRegimeRule] = field(default_factory=dict)


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
        IndicatorSpec(
            "drawdown", +1.0
        ),  # drawdown is negative; "less negative" is better
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


def _sigmoid(x: pd.Series | float) -> pd.Series | float:
    return 1.0 / (1.0 + np.exp(-x))


def _pick_mom_col(df: pd.DataFrame, day_label: int) -> str | None:
    """
    Prefer custom-horizon naming `mom_ret_{N}d`, fall back to default naming `mom_ret_{N}`.
    """
    d = int(day_label)
    c1 = f"mom_ret_{d}d"
    if c1 in df.columns:
        return c1
    c2 = f"mom_ret_{d}"
    if c2 in df.columns:
        return c2
    return None


def _groupwise_zscore(df: pd.DataFrame, col: str, keys: list[str]) -> pd.Series:
    if not keys:
        return _zscore(df[col]).astype("float64")
    return (
        df.groupby(keys, dropna=False, sort=False)[col]
        .transform(_zscore)
        .astype("float64")
    )


def apply_topdown_to_candidates(
    candidates: pd.DataFrame,
    *,
    date_col: str = "date",
    regime_col: str = "regime",
    evaluation: TopDownEvaluation | None = None,
) -> pd.DataFrame:
    """
    Apply a top-down, regime-aware multi-horizon scoring model.

    Adds:
    - `layer_z_quality_365`
    - `layer_z_structural_63_126`
    - `layer_z_confirm_30`
    - `layer_z_trigger_21`
    - `layer_z_micro_7`
    - `topdown_gate`
    - `edge_score_topdown`
    """
    if candidates.empty:
        out = candidates.copy()
        out["edge_score_topdown"] = pd.Series(dtype="float64")
        return out

    ev = evaluation or TopDownEvaluation()
    out = candidates.copy()

    keys: list[str] = []
    if date_col in out.columns:
        keys.append(date_col)
    if ev.regime_aware and regime_col in out.columns:
        keys.append(regime_col)

    _LOG.info(
        "topdown: apply_topdown_to_candidates: rows=%d keys=%s regime_aware=%s",
        len(out),
        keys,
        bool(ev.regime_aware),
    )

    # --- 365D "business quality" (structural / fundamental proxies)
    qcols = [c for c in ev.quality_cols if c in out.columns]
    if qcols:
        _LOG.info("topdown: quality_365: using cols=%s", qcols)
        out["_raw_quality_365"] = out[qcols].mean(axis=1)
        out["layer_z_quality_365"] = _groupwise_zscore(out, "_raw_quality_365", keys)
    else:
        _LOG.info(
            "topdown: quality_365: no quality cols present (expected one of=%s)",
            list(ev.quality_cols),
        )
        out["layer_z_quality_365"] = np.nan

    # --- Helper to build horizon layers from momentum columns
    def _build_mom_layer(days: tuple[int, ...], *, out_col: str) -> None:
        cols: list[str] = []
        for d in days:
            c = _pick_mom_col(out, d)
            if c is not None:
                cols.append(c)
        if not cols:
            _LOG.info(
                "topdown: %s: no momentum cols found for days=%s (expected naming: mom_ret_{N}d or mom_ret_{N})",
                out_col,
                list(days),
            )
            out[out_col] = np.nan
            return
        _LOG.info("topdown: %s: using cols=%s", out_col, cols)
        out[f"_raw_{out_col}"] = out[cols].mean(axis=1)
        out[out_col] = _groupwise_zscore(out, f"_raw_{out_col}", keys)

    _build_mom_layer(ev.structural_days, out_col="layer_z_structural_63_126")
    _build_mom_layer(ev.confirm_days, out_col="layer_z_confirm_30")
    _build_mom_layer(ev.trigger_days, out_col="layer_z_trigger_21")
    _build_mom_layer(ev.micro_days, out_col="layer_z_micro_7")

    # --- Regime-specific combination with top-down gating
    def _rule_for(reg: Any) -> TopDownRegimeRule:
        r = str(reg) if reg is not None else ""
        if r in ev.regime_rules:
            return ev.regime_rules[r]
        # Small convenience defaults for common labels used in this repo/tests.
        if r == "market_risk_on":
            return TopDownRegimeRule(
                weight_quality_365=0.75,
                weight_structural_63_126=1.0,
                weight_confirm_30=1.0,
                weight_trigger_21=1.0,
                weight_micro_7=0.5,
                micro_mode="pullback",
                gate_strength=1.25,
            )
        if r == "market_risk_off":
            return TopDownRegimeRule(
                weight_quality_365=1.25,
                weight_structural_63_126=0.5,
                weight_confirm_30=0.0,
                weight_trigger_21=0.0,
                weight_micro_7=0.0,
                micro_mode="ignore",
                gate_strength=1.0,
            )
        return ev.default_rule

    # Build per-row rules efficiently.
    if ev.regime_aware and regime_col in out.columns:
        rules = out[regime_col].map(_rule_for)
    else:
        rules = pd.Series([ev.default_rule] * len(out), index=out.index, dtype="object")

    # Compact rule summary (useful for confirming which regime rules are active).
    if ev.regime_aware and regime_col in out.columns:
        try:
            vc = out[regime_col].astype("string").value_counts(dropna=False)
            for reg, n in vc.items():
                rr = _rule_for(reg)
                _LOG.info(
                    "topdown: regime=%s n=%d rule=(wq=%.3f ws=%.3f wc=%.3f wt=%.3f wm=%.3f micro_mode=%s gate_strength=%.3f)",
                    str(reg),
                    int(n),
                    float(rr.weight_quality_365),
                    float(rr.weight_structural_63_126),
                    float(rr.weight_confirm_30),
                    float(rr.weight_trigger_21),
                    float(rr.weight_micro_7),
                    str(rr.micro_mode),
                    float(rr.gate_strength),
                )
        except Exception:
            # Best-effort only; scoring should still proceed if summarization fails.
            pass

    q = pd.to_numeric(out["layer_z_quality_365"], errors="coerce").astype("float64")
    s = pd.to_numeric(out["layer_z_structural_63_126"], errors="coerce").astype("float64")
    c = pd.to_numeric(out["layer_z_confirm_30"], errors="coerce").astype("float64")
    t = pd.to_numeric(out["layer_z_trigger_21"], errors="coerce").astype("float64")
    m = pd.to_numeric(out["layer_z_micro_7"], errors="coerce").astype("float64")

    # Micro timing adjustment (pullback vs follow).
    micro_adj = m.copy()
    micro_mode = rules.map(lambda rr: rr.micro_mode)
    micro_adj = micro_adj.where(micro_mode != "pullback", other=-micro_adj)
    micro_adj = micro_adj.where(micro_mode != "ignore", other=0.0)

    # Soft gate: product of sigmoid(long layers). This makes short-horizon layers matter
    # most when long-horizon context is supportive, without hard filtering.
    gs = rules.map(lambda rr: float(rr.gate_strength)).astype("float64")
    gate = (
        _sigmoid(gs * q).astype("float64")
        * _sigmoid(gs * s).astype("float64")
        * _sigmoid(gs * c).astype("float64")
    )
    out["topdown_gate"] = gate
    try:
        # Quick health check for gating behavior.
        g = pd.to_numeric(gate, errors="coerce").astype("float64")
        nn = int(g.notna().sum())
        _LOG.info(
            "topdown: gate: non_nan=%d/%d min=%.6f mean=%.6f max=%.6f",
            nn,
            len(g),
            float(g.min()) if nn else float("nan"),
            float(g.mean()) if nn else float("nan"),
            float(g.max()) if nn else float("nan"),
        )
    except Exception:
        pass

    wq = rules.map(lambda rr: float(rr.weight_quality_365)).astype("float64")
    ws = rules.map(lambda rr: float(rr.weight_structural_63_126)).astype("float64")
    wc = rules.map(lambda rr: float(rr.weight_confirm_30)).astype("float64")
    wt = rules.map(lambda rr: float(rr.weight_trigger_21)).astype("float64")
    wm = rules.map(lambda rr: float(rr.weight_micro_7)).astype("float64")

    # "Reverse order": trigger + micro are gated by the long-horizon context.
    out["edge_score_topdown"] = (wq * q) + (ws * s) + (wc * c) + gate * (
        (wt * t) + (wm * micro_adj)
    )
    try:
        sc = pd.to_numeric(out["edge_score_topdown"], errors="coerce").astype("float64")
        nn = int(sc.notna().sum())
        _LOG.info(
            "topdown: edge_score_topdown: non_nan=%d/%d min=%.6f mean=%.6f max=%.6f",
            nn,
            len(sc),
            float(sc.min()) if nn else float("nan"),
            float(sc.mean()) if nn else float("nan"),
            float(sc.max()) if nn else float("nan"),
        )
    except Exception:
        pass

    # Cleanup internal helper columns.
    drop_cols = [c for c in out.columns if c.startswith("_raw_")]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


def apply_indicators_to_candidates(
    candidates: pd.DataFrame,
    *,
    date_col: str = "date",
    regime_col: str = "regime",
    evaluation: EdgeEvaluation | TopDownEvaluation | None = None,
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

    # Optional alternate strategy: top-down horizon integration.
    if isinstance(evaluation, TopDownEvaluation):
        _LOG.info(
            "apply_indicators_to_candidates: strategy=topdown_multi_horizon rows=%d",
            len(candidates),
        )
        out = apply_topdown_to_candidates(
            candidates, date_col=date_col, regime_col=regime_col, evaluation=evaluation
        )
        # Keep `edge_score` stable for downstream consumers.
        out["edge_score"] = pd.to_numeric(
            out["edge_score_topdown"], errors="coerce"
        ).astype("float64")
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
    df = (
        df.dropna(subset=[symbol_col, date_col])
        .sort_values([symbol_col, date_col])
        .reset_index(drop=True)
    )

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
    ohlcv_table: str = "core.prices_daily",
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
    # market regime (DB)
    market_regime_table: str = "market_regime_daily",
    market_regime_date_col: str = "analysis_date",
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
        regime_summary_msg: str | None = None
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
        try:
            baseline = fetch_sp500_baseline(
                c,
                table=baseline_table,
                symbol_col=baseline_symbol_col,
                as_of_col=baseline_as_of_col,
                as_of=baseline_as_of,
            )
        except Exception as exc:
            # If the baseline membership table doesn't exist (common when you only
            # have OHLCV loaded), infer the universe from the OHLCV table instead.
            attempted_tables = {str(baseline_table)}
            if str(baseline_table) == "sp500":
                attempted_tables.add("sp500_tickers")
            if any(_missing_table_error(exc, table_name=t) for t in attempted_tables):
                _LOG.warning(
                    "Baseline table missing (%s). Falling back to DISTINCT symbols from %s.",
                    ", ".join(sorted(attempted_tables)),
                    ohlcv_table,
                )
                baseline = fetch_symbol_universe_from_table(
                    c,
                    table=ohlcv_table,
                    symbol_col=ohlcv_symbol_col,
                    out_col=baseline_symbol_col,
                )
            else:
                raise
        symbols = baseline[baseline_symbol_col].astype("string").dropna().tolist()
        _LOG.info(
            "run_edge: baseline rows=%d (table=%s)", len(baseline), baseline_table
        )
        _LOG.info("run_edge: baseline symbols=%d", len(symbols))
        if _LOG.isEnabledFor(logging.DEBUG):
            _LOG.debug("run_edge: baseline symbol list=%s", symbols)
        else:
            preview = symbols[:_LOG_SYMBOLS_PREVIEW_LIMIT]
            suffix = (
                ""
                if len(symbols) <= _LOG_SYMBOLS_PREVIEW_LIMIT
                else f" ... (+{len(symbols) - len(preview)} more)"
            )
            _LOG.info("run_edge: baseline symbol preview=%s%s", preview, suffix)
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
        _LOG.info(
            "run_edge: feature rows=%d cols=%d",
            len(feats),
            feats.shape[1] if not feats.empty else 0,
        )

        # Pull the canonical market regime (if available) and join on date.
        # This makes `regime` in the candidates reflect the market regime table,
        # while preserving per-symbol regime primitives (risk_on_off/vol_regime/regime_id).
        try:
            if not feats.empty and "date" in feats.columns:
                start_dt = pd.to_datetime(feats["date"], errors="coerce").min()
                end_dt = pd.to_datetime(feats["date"], errors="coerce").max()
                if pd.notna(start_dt) and pd.notna(end_dt):
                    mkt = fetch_market_regime_daily(
                        c,
                        table=market_regime_table,
                        analysis_date_col=market_regime_date_col,
                        start=start_dt,
                        end=end_dt,
                    )
                else:
                    mkt = fetch_market_regime_daily(
                        c,
                        table=market_regime_table,
                        analysis_date_col=market_regime_date_col,
                    )
                if not mkt.empty and market_regime_date_col in mkt.columns:
                    m = mkt.copy()
                    m[market_regime_date_col] = pd.to_datetime(
                        m[market_regime_date_col], errors="coerce"
                    )
                    m = m.dropna(subset=[market_regime_date_col]).sort_values(
                        market_regime_date_col
                    )
                    # Ensure exactly one row per day (if upstream accidentally has duplicates).
                    m = m.drop_duplicates(subset=[market_regime_date_col], keep="last")

                    # Summarize dominant regime over the last ~3 months plus latest trend.
                    # Best-effort: if anything looks off, we skip the summary and keep going.
                    try:
                        as_of = candidates_as_of
                        if as_of is None:
                            as_of = m[market_regime_date_col].max()
                        _, summary = summarize_market_regime(
                            m,
                            as_of=as_of,
                            lookback_days=90,
                            trend_lookback_days=20,
                            date_col=market_regime_date_col,
                            regime_col="regime_label",
                            score_col="regime_score",
                        )
                        share = (
                            (summary.dominant_days / float(summary.total_days))
                            if summary.total_days
                            else 0.0
                        )
                        share20 = (
                            float(summary.recent_share_latest)
                            if summary.recent_share_latest is not None
                            else 0.0
                        )
                        slope = summary.recent_score_slope
                        slope_str = (
                            f"{float(slope):+.6f}"
                            if slope is not None and np.isfinite(float(slope))
                            else "n/a"
                        )

                        # Multi-line, human-friendly summary for logs.
                        regime_summary_msg = (
                            "run_edge: market regime summary (last ~3 months)\n"
                            f"  as_of: {summary.as_of}\n"
                            f"  dominant: {summary.dominant_regime} ({summary.dominant_days}/{summary.total_days}={100.0 * share:.1f}%)\n"
                            f"  latest: {summary.latest_regime} (streak {summary.latest_streak_days}d, share_last_20d={100.0 * share20:.1f}%, score_slope_20d={slope_str})"
                        )
                    except Exception:
                        regime_summary_msg = None

                    feats2 = feats.copy()
                    feats2["date"] = pd.to_datetime(feats2["date"], errors="coerce")
                    merged = feats2.merge(
                        m,
                        left_on="date",
                        right_on=market_regime_date_col,
                        how="left",
                        suffixes=("", "_mkt"),
                    )
                    # Replace the derived per-symbol regime label with the market regime label when available.
                    if "regime_label" in merged.columns:
                        merged["regime"] = merged["regime_label"].astype("string")
                    feats = merged
                    _LOG.info(
                        "run_edge: joined market regime rows=%d cols=%d (table=%s)",
                        len(mkt),
                        len(mkt.columns),
                        market_regime_table,
                    )
        except Exception as exc:
            # If the market regime table isn't present (or has schema differences), keep the derived regime.
            _LOG.warning(
                "run_edge: market regime join skipped (%s). Using derived regime.", exc
            )

        candidates = build_candidate_list(
            feats, symbol_col="symbol", date_col="date", as_of=candidates_as_of
        )
        if regime_summary_msg:
            _LOG.info(
                "run_edge: candidates=%d\n%s", len(candidates), regime_summary_msg
            )
        else:
            _LOG.info("run_edge: candidates=%d", len(candidates))
        scored = apply_indicators_to_candidates(candidates, evaluation=evaluation)
        _LOG.info(
            "run_edge: scored candidates=%d (edge_score present=%s)",
            len(scored),
            "edge_score" in scored.columns,
        )
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


def _run_and_print_edge(
    conn: Any,
    *,
    args: argparse.Namespace,
    baseline_as_of_col: str | None,
    horizon_days: tuple[int, ...] | None,
) -> int:
    out = run_edge(
        conn=conn,
        baseline_table=args.baseline_table,
        baseline_symbol_col=args.baseline_symbol_col,
        baseline_as_of_col=baseline_as_of_col,  # type: ignore[arg-type]
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

    score_groups: Any = None
    if not args.no_score_performance:
        try:
            from .score_performance import fetch_score_name_groups

            score_groups = fetch_score_name_groups(
                conn, table=args.score_performance_table
            )
        except Exception as exc:
            _LOG.warning("score_performance: skipped (%s)", exc)

    shown = out if not args.top else out.head(int(args.top))
    if shown.empty:
        print("(no candidates)")
        return 0

    # Keep the CLI output compact and stable.
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 200
    ):
        print(shown.to_string(index=False))

    # `fetch_score_name_groups()` returns a DataFrame, which cannot be used in a boolean
    # context (pandas raises: "The truth value of a DataFrame is ambiguous.").
    # Historically this CLI printed a small summary when groups were available; keep the
    # behavior but make it robust to both DataFrame and dict-like returns.
    if score_groups is not None:
        if isinstance(score_groups, pd.DataFrame):
            if not score_groups.empty:
                print("\nScore performance groups:")
                with pd.option_context(
                    "display.max_rows",
                    None,
                    "display.max_columns",
                    None,
                    "display.width",
                    200,
                ):
                    print(score_groups.to_string(index=False))
        elif isinstance(score_groups, dict):
            if score_groups:
                print("\nScore performance groups:")
                for k in sorted(score_groups.keys()):
                    try:
                        n = len(score_groups[k])
                    except Exception:
                        n = 0
                    print(f"- {k}: {n} scores")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI entrypoint.

    Examples:
      - Demo (no DB required):
        python3 -m springedge.edge --demo

      - Score performance helper (separate CLI, routed through this module for convenience):
        python3 -m springedge.edge --score-performance --demo
        python3 -m springedge.score_performance --demo

      - Real DB:
        export SPRINGEDGE_DB_URL="postgresql://..."
        python3 -m springedge.edge --baseline-as-of 2024-02-01 --horizon-days 7 21
    """
    # Convenience dispatch: allow running `score_performance` via this CLI too.
    # This avoids a common confusion where users run `python3 -m springedge.edge`
    # and expect `--score-performance` to work (it historically only existed in the
    # repo-root `edge.py` wrapper).
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    if "--score-performance" in argv_list:
        from .score_performance import main as _sp_main

        forwarded = [a for a in argv_list if a != "--score-performance"]
        return int(_sp_main(forwarded))

    p = argparse.ArgumentParser(
        prog="springedge.edge",
        description="Run end-to-end Edge orchestration and print scored candidates.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING). Default: INFO.",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Run a self-contained sqlite demo (no external DB required).",
    )
    p.add_argument(
        "--score-performance",
        action="store_true",
        help="Run the score_performance CLI instead of the Edge pipeline.",
    )
    p.add_argument(
        "--no-score-performance",
        action="store_true",
        help="Do not attempt to fetch/print score performance groups after the Edge run.",
    )
    p.add_argument(
        "--score-performance-table",
        default="score_performance_evaluation",
        help="Score performance source table. Default: score_performance_evaluation (also tries intelligence.score_performance_evaluation).",
    )
    p.add_argument(
        "--db-url",
        default=None,
        help="Optional DB URL override (otherwise uses env var SPRINGEDGE_DB_URL or DATABASE_URL).",
    )
    p.add_argument(
        "--db-env-var",
        default="SPRINGEDGE_DB_URL",
        help="Environment variable to read DB URL from. Default: SPRINGEDGE_DB_URL (also falls back to DATABASE_URL).",
    )

    # Baseline / layers
    p.add_argument(
        "--baseline-table",
        default="sp500",
        help="Baseline universe table. Default: sp500.",
    )
    p.add_argument(
        "--baseline-symbol-col",
        default="symbol",
        help="Baseline symbol column. Default: symbol.",
    )
    p.add_argument(
        "--baseline-as-of-col",
        default="date",
        help="Baseline 'as of' date column (use ''/none to disable). Default: date.",
    )
    p.add_argument(
        "--baseline-as-of",
        default=None,
        help="Baseline as-of date filter (inclusive). Default: latest snapshot.",
    )

    # OHLCV
    p.add_argument(
        "--ohlcv-table",
        default="core.prices_daily",
        help="OHLCV source table. Default: core.prices_daily.",
    )
    p.add_argument("--ohlcv-symbol-col", default="symbol")
    p.add_argument("--ohlcv-date-col", default="date")
    p.add_argument(
        "--ohlcv-start", default=None, help="OHLCV start date filter (inclusive)."
    )
    p.add_argument(
        "--ohlcv-end", default=None, help="OHLCV end date filter (inclusive)."
    )

    # Features / evaluation
    p.add_argument("--universe", default="sp500")
    p.add_argument(
        "--horizon-days",
        nargs="*",
        default=None,
        help="Horizon days (e.g. `--horizon-days 7 21` or `--horizon-days 7,21`).",
    )
    p.add_argument(
        "--horizon-basis", choices=("trading", "calendar"), default="trading"
    )
    p.add_argument(
        "--candidates-as-of",
        default=None,
        help="Build one row per symbol as-of this date.",
    )
    p.add_argument(
        "--top",
        type=int,
        default=0,
        help="How many candidates to print (0 = all). Default: 0.",
    )
    # Parse from the already-normalized argv_list to keep behavior consistent for both
    # `argv is None` and explicit `argv`.
    args = p.parse_args(list(argv_list))

    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    configure_logging(level=level)

    horizon_days = _parse_horizon_days(args.horizon_days)

    baseline_as_of_col = str(args.baseline_as_of_col or "").strip()
    if baseline_as_of_col.lower() in {"", "none", "null"}:
        baseline_as_of_col = None  # type: ignore[assignment]

    if args.demo:
        import sqlite3

        conn = sqlite3.connect(":memory:")
        # sqlite schema-qualification requires ATTACH for `core.<table>` style names.
        conn.execute("ATTACH DATABASE ':memory:' AS core")
        conn.execute("create table sp500 (date text, symbol text)")
        conn.execute(
            "create table core.prices_daily (symbol text, date text, open real, high real, low real, close real, volume real)"
        )
        conn.execute(
            """
            create table market_regime_daily (
              analysis_date text,
              regime_label text,
              regime_score real,
              position_multiplier real,
              vix_level real,
              vix_zscore real,
              vix9d_over_vix real,
              vix_vix3m_ratio real,
              move_level real,
              move_zscore real,
              term_structure_state text,
              event_risk_flag integer,
              joint_stress_flag integer,
              notes text,
              created_at text
            )
            """
        )

        # Baseline snapshot (latest = 2024-02-01).
        conn.executemany(
            "insert into sp500 (date, symbol) values (?, ?)",
            [
                ("2024-01-02", "AAA"),
                ("2024-01-02", "BBB"),
                ("2024-02-01", "AAA"),
                ("2024-02-01", "BBB"),
            ],
        )

        # Simple synthetic OHLCV for both symbols.
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
                "insert into core.prices_daily (symbol, date, open, high, low, close, volume) values (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

        # Provide a market regime row for the last OHLCV date so the pipeline can join it.
        last_date = dates[-1].date().isoformat()
        conn.execute(
            """
            insert into market_regime_daily (
              analysis_date, regime_label, regime_score, position_multiplier,
              vix_level, vix_zscore, vix9d_over_vix, vix_vix3m_ratio,
              move_level, move_zscore, term_structure_state, event_risk_flag,
              joint_stress_flag, notes, created_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                last_date,
                "market_risk_on",
                1.23,
                1.0,
                15.0,
                0.1,
                0.9,
                0.8,
                100.0,
                -0.2,
                "normal",
                0,
                0,
                None,
                last_date,
            ),
        )

        # Demo defaults.
        if args.baseline_as_of is None:
            args.baseline_as_of = "2024-02-01"
        if horizon_days is None:
            horizon_days = (7, 21)

        try:
            return _run_and_print_edge(
                conn,
                args=args,
                baseline_as_of_col=baseline_as_of_col,
                horizon_days=horizon_days,
            )
        finally:
            try:
                conn.close()
            except Exception:
                pass

    with db_connection(args.db_url, env_var=args.db_env_var) as conn:
        return _run_and_print_edge(
            conn,
            args=args,
            baseline_as_of_col=baseline_as_of_col,
            horizon_days=horizon_days,
        )


if __name__ == "__main__":
    raise SystemExit(main())
