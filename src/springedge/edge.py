# ruff: noqa: E402
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
import json
import logging
import re
import sys
from dataclasses import asdict, is_dataclass
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


@dataclass(frozen=True)
class SignalLayerPolicyRow:
    signal_name: str
    horizon_days: int
    layer_name: str  # quality_365 | confirm_30 | micro_7 | ...
    weight: float
    min_regime: str = "ANY"  # CAUTION | POSITIVE | ANY
    enabled: bool = True


@dataclass(frozen=True)
class HorizonThresholds:
    spearman_ic: float
    ic_ir: float
    q5_minus_q1: float
    n_obs: int
    # Optional extras
    reject_if_ic_ir_below: float | None = None
    reject_if_sign_flip_across_regimes: bool = False
    strong_ic_ir: float | None = None
    strong_q5_minus_q1: float | None = None


_HORIZON_THRESHOLDS: dict[int, HorizonThresholds] = {
    # STEP 1 (non-negotiable): exact thresholds per horizon
    # 7D micro / timing
    7: HorizonThresholds(
        spearman_ic=0.03,
        ic_ir=0.8,
        q5_minus_q1=0.3,
        n_obs=500,
        reject_if_ic_ir_below=0.5,
        reject_if_sign_flip_across_regimes=True,
    ),
    # 21–30D confirmation / tactical (use the same thresholds for both 21 and 30)
    21: HorizonThresholds(
        spearman_ic=0.05,
        ic_ir=1.2,
        q5_minus_q1=0.7,
        n_obs=1_000,
        strong_ic_ir=1.8,
        strong_q5_minus_q1=1.2,
    ),
    30: HorizonThresholds(
        spearman_ic=0.05,
        ic_ir=1.2,
        q5_minus_q1=0.7,
        n_obs=1_000,
        strong_ic_ir=1.8,
        strong_q5_minus_q1=1.2,
    ),
    # 90–180D structural
    90: HorizonThresholds(spearman_ic=0.08, ic_ir=1.5, q5_minus_q1=1.0, n_obs=2_000),
    180: HorizonThresholds(
        spearman_ic=0.08, ic_ir=1.5, q5_minus_q1=1.0, n_obs=2_000
    ),
    # 365D quality / ownership
    365: HorizonThresholds(spearman_ic=0.10, ic_ir=2.0, q5_minus_q1=2.0, n_obs=2_000),
}


def _safe_float(x: object) -> float | None:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return None
    return v if np.isfinite(v) else None


def _make_cli_formatters(df: pd.DataFrame) -> dict[str, Any]:
    """
    Provide stable, readable number formatting for CLI tables.

    We keep this intentionally conservative (only applies to well-known columns) so it
    doesn't unexpectedly change formatting for user-defined/custom columns.
    """

    def _fmt_int(x: object) -> str:
        v = _safe_float(x)
        if v is None:
            return ""
        return f"{int(round(v)):,}"

    def _fmt_2(x: object) -> str:
        v = _safe_float(x)
        if v is None:
            return ""
        return f"{v:,.2f}"

    def _fmt_4(x: object) -> str:
        v = _safe_float(x)
        if v is None:
            return ""
        return f"{v:,.4f}"

    def _fmt_6(x: object) -> str:
        v = _safe_float(x)
        if v is None:
            return ""
        return f"{v:,.6f}"

    def _fmt_sci(x: object) -> str:
        v = _safe_float(x)
        if v is None:
            return ""
        return f"{v:.3e}"

    fmts: dict[str, Any] = {}

    # Common price/liquidity fields.
    for c in ("close", "open", "high", "low", "vix_level", "move_level"):
        if c in df.columns:
            fmts[c] = _fmt_2
    for c in ("volume",):
        if c in df.columns:
            fmts[c] = _fmt_int
    for c in ("liquidity",):
        if c in df.columns:
            fmts[c] = _fmt_sci

    # Regime context metrics.
    for c in ("regime_score", "position_multiplier"):
        if c in df.columns:
            fmts[c] = _fmt_2
    for c in ("vix_zscore", "move_zscore", "vix9d_over_vix", "vix_vix3m_ratio"):
        if c in df.columns:
            fmts[c] = _fmt_4

    # Scores: keep precision (these are often used for ranking/debugging).
    for c in ("edge_score", "edge_score_topdown", "topdown_gate"):
        if c in df.columns:
            fmts[c] = _fmt_6

    return fmts


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


def _is_sqlite_conn(conn: Any) -> bool:
    mod = str(getattr(conn.__class__, "__module__", "") or "")
    return "sqlite3" in mod


def _json_dumps_safe(obj: Any) -> str:
    """
    Best-effort JSON serializer for dataclasses + numpy scalars.
    """

    def _default(o: Any) -> Any:
        if is_dataclass(o):
            return asdict(o)
        # numpy scalars
        if hasattr(o, "item"):
            try:
                return o.item()
            except Exception:
                pass
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    return json.dumps(obj, default=_default, sort_keys=True)


def ensure_topdown_evaluation_tables(
    conn: Any,
    *,
    run_table: str = "topdown_evaluation_run",
    result_table: str = "topdown_evaluation_result",
) -> None:
    """
    Create tables to persist `TopDownEvaluation` outcomes (portable DB-API DDL).

    Tables:
    - run_table: one row per evaluation run (params/config JSON)
    - result_table: one row per (run_id, symbol) result with top-down layer scores
    """
    rt = _validate_table_ref(run_table, kind="table")
    res_t = _validate_table_ref(result_table, kind="table")

    cur = conn.cursor()
    try:
        if _is_sqlite_conn(conn):
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {rt} (
                  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  evaluation_json TEXT,
                  params_json TEXT,
                  notes TEXT
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {res_t} (
                  run_id INTEGER NOT NULL,
                  symbol TEXT NOT NULL,
                  as_of_date TEXT NOT NULL,
                  regime TEXT,
                  edge_score_topdown REAL,
                  topdown_gate REAL,
                  layer_z_quality_365 REAL,
                  layer_z_structural_63_126 REAL,
                  layer_z_confirm_30 REAL,
                  layer_z_trigger_21 REAL,
                  layer_z_micro_7 REAL,
                  extras_json TEXT,
                  PRIMARY KEY (run_id, symbol),
                  FOREIGN KEY (run_id) REFERENCES {rt}(run_id)
                )
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{result_table.split('.')[-1]}_run_id ON {res_t}(run_id)"
            )
        else:
            # Postgres-ish (psycopg/psycopg2) DDL.
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {rt} (
                  run_id BIGSERIAL PRIMARY KEY,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                  evaluation_json JSONB,
                  params_json JSONB,
                  notes TEXT
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {res_t} (
                  run_id BIGINT NOT NULL REFERENCES {rt}(run_id),
                  symbol TEXT NOT NULL,
                  as_of_date DATE NOT NULL,
                  regime TEXT,
                  edge_score_topdown DOUBLE PRECISION,
                  topdown_gate DOUBLE PRECISION,
                  layer_z_quality_365 DOUBLE PRECISION,
                  layer_z_structural_63_126 DOUBLE PRECISION,
                  layer_z_confirm_30 DOUBLE PRECISION,
                  layer_z_trigger_21 DOUBLE PRECISION,
                  layer_z_micro_7 DOUBLE PRECISION,
                  extras_json JSONB,
                  PRIMARY KEY (run_id, symbol)
                )
                """
            )
            idx_name = f"idx_{result_table.split('.')[-1]}_run_id"
            cur.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {res_t}(run_id)")
    finally:
        try:
            cur.close()
        except Exception:
            pass

    try:
        if hasattr(conn, "commit"):
            conn.commit()
    except Exception:
        pass


def persist_topdown_evaluation(
    conn: Any,
    scored_candidates: pd.DataFrame,
    *,
    evaluation: TopDownEvaluation,
    params: dict[str, Any] | None = None,
    run_table: str = "topdown_evaluation_run",
    result_table: str = "topdown_evaluation_result",
    notes: str | None = None,
) -> int:
    """
    Persist `TopDownEvaluation` output rows and return the created run_id.
    """
    ensure_topdown_evaluation_tables(
        conn, run_table=run_table, result_table=result_table
    )

    rt = _validate_table_ref(run_table, kind="table")
    res_t = _validate_table_ref(result_table, kind="table")
    ph = _conn_placeholder(conn)

    missing = [c for c in ("symbol", "date") if c not in scored_candidates.columns]
    if missing:
        raise ValueError(f"scored_candidates missing required columns: {missing}")

    df = scored_candidates.copy()
    # Normalize types for DB-API drivers (avoid pandas Timestamp objects).
    df["symbol"] = df["symbol"].astype("string")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    evaluation_payload = asdict(evaluation)
    params_payload = dict(params or {})

    cur = conn.cursor()
    try:
        if _is_sqlite_conn(conn):
            cur.execute(
                f"""
                INSERT INTO {rt} (evaluation_json, params_json, notes)
                VALUES ({ph}, {ph}, {ph})
                """,
                (
                    _json_dumps_safe(evaluation_payload),
                    _json_dumps_safe(params_payload),
                    str(notes) if notes is not None else None,
                ),
            )
            run_id = int(getattr(cur, "lastrowid", 0) or 0)
        else:
            cur.execute(
                f"""
                INSERT INTO {rt} (evaluation_json, params_json, notes)
                VALUES ({ph}::jsonb, {ph}::jsonb, {ph})
                RETURNING run_id
                """,
                (
                    _json_dumps_safe(evaluation_payload),
                    _json_dumps_safe(params_payload),
                    str(notes) if notes is not None else None,
                ),
            )
            row = cur.fetchone()
            run_id = int(row[0]) if row else 0

        if not run_id:
            raise RuntimeError("Failed to create topdown evaluation run_id")

        # Optional extra context fields (stored as JSON to keep schema stable).
        extra_cols = [
            c
            for c in (
                "regime_id",
                "risk_on_off",
                "vol_regime",
                # Top-down missingness flags (kept in extras_json to avoid schema churn).
                "topdown_missing_quality_365",
                "topdown_missing_layer_z_structural_63_126",
                "topdown_missing_layer_z_confirm_30",
                "topdown_missing_layer_z_trigger_21",
                "topdown_missing_layer_z_micro_7",
            )
            if c in df.columns
        ]

        rows: list[tuple[Any, ...]] = []
        for r in df.itertuples(index=False):
            # Use getattr to tolerate optional columns.
            sym = str(getattr(r, "symbol"))
            dt = getattr(r, "date")
            as_of = (
                dt.date().isoformat()
                if isinstance(dt, datetime)
                else (dt.date().isoformat() if hasattr(dt, "date") else str(dt))
            )
            regime = getattr(r, "regime", None)
            extras: dict[str, Any] = {}
            for c in extra_cols:
                extras[c] = getattr(r, c, None)
            extras_json = _json_dumps_safe(extras) if extras else None

            def _f(name: str) -> float | None:
                if name not in df.columns:
                    return None
                v = getattr(r, name)
                try:
                    fv = float(v)
                    return fv if np.isfinite(fv) else None
                except Exception:
                    return None

            rows.append(
                (
                    int(run_id),
                    sym,
                    as_of,
                    str(regime) if regime is not None else None,
                    _f("edge_score_topdown"),
                    _f("topdown_gate"),
                    _f("layer_z_quality_365"),
                    _f("layer_z_structural_63_126"),
                    _f("layer_z_confirm_30"),
                    _f("layer_z_trigger_21"),
                    _f("layer_z_micro_7"),
                    extras_json,
                )
            )

        cols_sql = """
          run_id,
          symbol,
          as_of_date,
          regime,
          edge_score_topdown,
          topdown_gate,
          layer_z_quality_365,
          layer_z_structural_63_126,
          layer_z_confirm_30,
          layer_z_trigger_21,
          layer_z_micro_7,
          extras_json
        """
        extras_expr = f"{ph}" if _is_sqlite_conn(conn) else f"{ph}::jsonb"
        insert_sql = (
            f"INSERT INTO {res_t} ({cols_sql}) VALUES "
            f"({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {extras_expr})"
        )

        cur.executemany(insert_sql, rows)
    finally:
        try:
            cur.close()
        except Exception:
            pass

    try:
        if hasattr(conn, "commit"):
            conn.commit()
    except Exception:
        pass
    return int(run_id)


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


def ensure_signal_layer_policy_table(
    conn: Any,
    *,
    table: str = "intelligence.signal_layer_policy",
) -> str:
    """
    STEP 3: create the missing backbone table (portable DB-API DDL).

    Returns the table reference actually created/used (may differ on sqlite if the
    requested schema isn't attached).
    """
    requested = str(table)
    # Validate identifiers (schema-qualified allowed).
    try:
        t = _validate_table_ref(requested, kind="table")
    except Exception:
        # If callers pass something odd, fall back to a safe default.
        t = "signal_layer_policy"

    cur = conn.cursor()
    try:
        if _is_sqlite_conn(conn):
            # sqlite supports schema-qualified names only when the schema is ATTACHed.
            # If it isn't, fallback to an unqualified table name.
            try:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {t} (
                      signal_name TEXT,
                      horizon_days INT,
                      layer_name TEXT,
                      weight FLOAT,
                      min_regime TEXT DEFAULT 'ANY',
                      enabled BOOLEAN DEFAULT TRUE,
                      PRIMARY KEY (signal_name, horizon_days, layer_name)
                    )
                    """
                )
                out_table = t
            except Exception as exc:
                # "unknown database intelligence" / similar
                fallback = "signal_layer_policy"
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {fallback} (
                      signal_name TEXT,
                      horizon_days INT,
                      layer_name TEXT,
                      weight FLOAT,
                      min_regime TEXT DEFAULT 'ANY',
                      enabled BOOLEAN DEFAULT TRUE,
                      PRIMARY KEY (signal_name, horizon_days, layer_name)
                    )
                    """
                )
                _LOG.info(
                    "signal_layer_policy: sqlite schema not attached; using table=%s (requested=%s, err=%s)",
                    fallback,
                    requested,
                    exc,
                )
                out_table = fallback
        else:
            # Postgres-ish: ensure schema exists if schema-qualified.
            if "." in t:
                schema = t.split(".")[0]
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {t} (
                  signal_name TEXT,
                  horizon_days INT,
                  layer_name TEXT,
                  weight DOUBLE PRECISION,
                  min_regime TEXT DEFAULT 'ANY',
                  enabled BOOLEAN DEFAULT TRUE,
                  PRIMARY KEY (signal_name, horizon_days, layer_name)
                )
                """
            )
            out_table = t
    finally:
        try:
            cur.close()
        except Exception:
            pass

    try:
        if hasattr(conn, "commit"):
            conn.commit()
    except Exception:
        pass
    return str(out_table)


def default_sp500_signal_layer_policy_rows() -> list[SignalLayerPolicyRow]:
    """
    STEP 2 + STEP 3: production-grade whitelist expressed as policy rows.
    """
    rows: list[SignalLayerPolicyRow] = []

    # 🟣 Quality Layer (365D)
    for s in [
        "nonuple.growth",
        "nonuple.competitive",
        "nonuple.financial",
        "nonuple.profitability",
        "fin.revenue_growth",
        "fin.roe",
    ]:
        rows.append(
            SignalLayerPolicyRow(
                signal_name=s,
                horizon_days=365,
                layer_name="quality_365",
                weight=1.0,
                min_regime="ANY",
                enabled=True,
            )
        )

    # 🟡 Confirmation Layer (21–30D) — use 30 as the canonical horizon label.
    rows.extend(
        [
            SignalLayerPolicyRow(
                "trs_combined.tqi_current", 30, "confirm_30", 1.2, "ANY", True
            ),
            SignalLayerPolicyRow("trs_combined.total", 30, "confirm_30", 1.0, "ANY", True),
            SignalLayerPolicyRow("mqps.tape_quality", 30, "confirm_30", 1.0, "ANY", True),
            SignalLayerPolicyRow(
                "mqps.days_since_earnings", 30, "confirm_30", 1.0, "ANY", True
            ),
            SignalLayerPolicyRow(
                "options_flow.institutional_flow", 30, "confirm_30", 1.0, "ANY", True
            ),
            # Only include in CAUTION regimes (per user example).
            SignalLayerPolicyRow(
                "options_flow.unusual_activity", 30, "confirm_30", 0.8, "CAUTION", True
            ),
        ]
    )

    # 🟢 Micro Layer (7D)
    for s in [
        "options_flow.momentum",
        "options_flow.large_block_activity",
        "tech.volume_confirmation_score",
        "trs_combined.volume_ratio_20d",
    ]:
        rows.append(
            SignalLayerPolicyRow(
                signal_name=s,
                horizon_days=7,
                layer_name="micro_7",
                weight=0.5,
                min_regime="ANY",
                enabled=True,
            )
        )

    return rows


def upsert_signal_layer_policy(
    conn: Any,
    rows: Sequence[SignalLayerPolicyRow],
    *,
    table: str = "intelligence.signal_layer_policy",
) -> str:
    """
    Insert/update rows into `signal_layer_policy`.

    Returns the table reference actually used (sqlite may fallback to unqualified).
    """
    t = ensure_signal_layer_policy_table(conn, table=table)
    ph = _conn_placeholder(conn)

    cols = "signal_name, horizon_days, layer_name, weight, min_regime, enabled"
    if _is_sqlite_conn(conn):
        sql = f"INSERT OR REPLACE INTO {t} ({cols}) VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph})"
    else:
        sql = f"""
        INSERT INTO {t} ({cols})
        VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph})
        ON CONFLICT (signal_name, horizon_days, layer_name)
        DO UPDATE SET
          weight = EXCLUDED.weight,
          min_regime = EXCLUDED.min_regime,
          enabled = EXCLUDED.enabled
        """

    data: list[tuple[Any, ...]] = []
    for r in rows:
        data.append(
            (
                str(r.signal_name),
                int(r.horizon_days),
                str(r.layer_name),
                float(r.weight),
                str(r.min_regime),
                bool(r.enabled),
            )
        )

    cur = conn.cursor()
    try:
        cur.executemany(sql, data)
    finally:
        try:
            cur.close()
        except Exception:
            pass
    try:
        if hasattr(conn, "commit"):
            conn.commit()
    except Exception:
        pass
    return str(t)


def _load_score_performance_df(
    conn: Any,
    *,
    table: str = "score_performance_evaluation",
) -> pd.DataFrame:
    """
    Best-effort loader for score performance stats, with common schema fallback.
    """
    def _fetch(_table: str) -> pd.DataFrame:
        t = _validate_table_ref(_table, kind="table")
        cur = conn.cursor()
        try:
            try:
                cur.execute(f"SELECT * FROM {t}")
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
        return pd.DataFrame.from_records(rows, columns=cols)

    try:
        return _fetch(table)
    except Exception as exc:
        if not _missing_table_error(exc, table_name=table):
            raise
        fallback = (
            f"intelligence.{table}" if "." not in str(table) else str(table).split(".")[-1]
        )
        try:
            return _fetch(fallback)
        except Exception as exc2:
            if _missing_table_error(exc2, table_name=fallback):
                return pd.DataFrame()
            raise


def evaluate_actionable_signals_by_horizon(
    conn: Any,
    *,
    horizon_days: int,
    table: str = "score_performance_evaluation",
) -> pd.DataFrame:
    """
    STEP 1: apply the exact horizon thresholds (non-negotiable).

    Returns a DataFrame with one row per score_name:
      [score_name, horizon_days, passed, strong, reason]
    """
    th = _HORIZON_THRESHOLDS.get(int(horizon_days))
    if th is None:
        raise ValueError(f"Unsupported horizon_days={horizon_days}; no thresholds defined.")

    df = _load_score_performance_df(conn, table=table)
    if df.empty:
        return pd.DataFrame(
            columns=["score_name", "horizon_days", "passed", "strong", "reason"]
        )

    df.columns = [str(c) for c in df.columns]
    cols = list(df.columns)

    def _first_present(candidates: Sequence[str]) -> str | None:
        s = set(cols)
        for cand in candidates:
            if cand in s:
                return cand
        return None

    score_col = _first_present(("score_name", "score", "name"))
    horizon_col = _first_present(("horizon_days", "horizon", "horizon_day", "horizon_d"))
    regime_col = _first_present(("regime_label", "regime", "market_regime", "regime_name"))
    ic_col = _first_present(("spearman_ic", "rank_ic", "ic_spearman", "spearman_rank_ic", "ic"))
    ir_col = _first_present(("ic_ir", "icir", "ir", "information_ratio"))
    spread_col = _first_present(("q5_minus_q1", "q5_q1", "q5_q1_spread", "q5q1"))
    nobs_col = _first_present(("n_obs", "sample_size", "n", "n_samples", "count"))

    missing: list[str] = []
    if score_col is None:
        missing.append("score_name")
    if horizon_col is None:
        missing.append("horizon_days")
    if ic_col is None:
        missing.append("spearman_ic")
    if ir_col is None:
        missing.append("ic_ir")
    if spread_col is None:
        missing.append("q5_minus_q1")
    if nobs_col is None:
        missing.append("n_obs")
    if missing:
        raise ValueError(f"score_performance table missing required columns: {missing}")

    w = df.copy()
    w[score_col] = w[score_col].astype("string")
    w[horizon_col] = pd.to_numeric(w[horizon_col], errors="coerce").astype("Int64")
    w = w.dropna(subset=[score_col, horizon_col]).copy()
    w = w[w[horizon_col] == int(horizon_days)].copy()
    if w.empty:
        return pd.DataFrame(
            columns=["score_name", "horizon_days", "passed", "strong", "reason"]
        )

    ic = pd.to_numeric(w[ic_col], errors="coerce").astype("float64")
    ir = pd.to_numeric(w[ir_col], errors="coerce").astype("float64")
    sp = pd.to_numeric(w[spread_col], errors="coerce").astype("float64")
    nobs = pd.to_numeric(w[nobs_col], errors="coerce").astype("float64")

    # Per-row pass (treat as "per regime row" when a regime column exists).
    row_pass = (
        (ic >= float(th.spearman_ic))
        & (ir >= float(th.ic_ir))
        & (sp >= float(th.q5_minus_q1))
        & (nobs >= float(th.n_obs))
    )

    # Optional explicit reject rules.
    if th.reject_if_ic_ir_below is not None:
        row_pass = row_pass & (ir >= float(th.reject_if_ic_ir_below))

    w["_row_pass"] = row_pass.fillna(False)

    # Sign flip across regimes (reject). If no regime column, treat as stable.
    if bool(th.reject_if_sign_flip_across_regimes) and regime_col is not None:
        signs = np.sign(ic.fillna(0.0))
        w["_ic_sign"] = signs
        gb_sign = w.groupby(score_col, dropna=False, sort=True)["_ic_sign"]
        # sign flip if there exist both positive and negative rows
        has_pos = gb_sign.apply(lambda s: bool((s > 0).any()))
        has_neg = gb_sign.apply(lambda s: bool((s < 0).any()))
        flip = (has_pos & has_neg).rename("_sign_flip").reset_index()
        w = w.merge(flip, on=score_col, how="left")
        w["_sign_flip"] = w["_sign_flip"].fillna(False)
        w["_row_pass"] = w["_row_pass"] & (~w["_sign_flip"])
    else:
        w["_sign_flip"] = False

    # Aggregate to score_name: require all rows (regimes) to pass.
    gb = w.groupby(score_col, dropna=False, sort=True)
    passed = gb["_row_pass"].all()

    # Strong flag (only relevant where thresholds define it).
    strong = pd.Series([False] * len(passed), index=passed.index, dtype="bool")
    if th.strong_ic_ir is not None and th.strong_q5_minus_q1 is not None:
        row_strong = (ir >= float(th.strong_ic_ir)) & (sp >= float(th.strong_q5_minus_q1))
        w["_row_strong"] = row_strong.fillna(False)
        strong = gb["_row_strong"].all()

    # Reason summary (compact; first failing reason).
    reasons: list[dict[str, Any]] = []
    for name in passed.index.astype(str).tolist():
        ok = bool(passed.loc[name])
        if ok:
            reason = ""
        else:
            # Find any failing row for this score.
            ww = w[w[score_col].astype(str) == str(name)]
            flip_any = bool(ww["_sign_flip"].fillna(False).any())
            if flip_any:
                reason = "IC flips sign across regimes"
            else:
                reason = "fails threshold(s)"
        reasons.append(
            {
                "score_name": str(name),
                "horizon_days": int(horizon_days),
                "passed": ok,
                "strong": bool(strong.loc[name]) if name in strong.index else False,
                "reason": reason,
            }
        )

    return pd.DataFrame.from_records(reasons)


def _market_tag_from_regime_row(regime_label: str | None, regime_score: float | None) -> str:
    """
    Map the current market regime to a simple policy tag: POSITIVE | CAUTION.
    """
    lbl = (str(regime_label or "")).lower()
    if regime_score is not None:
        try:
            if float(regime_score) < 0.0:
                return "CAUTION"
        except Exception:
            pass
    if "risk_off" in lbl:
        return "CAUTION"
    return "POSITIVE"


def _fetch_policy_rows(
    conn: Any,
    *,
    table: str = "intelligence.signal_layer_policy",
    only_enabled: bool = True,
) -> tuple[str, list[SignalLayerPolicyRow]]:
    t = ensure_signal_layer_policy_table(conn, table=table)
    cur = conn.cursor()
    try:
        if only_enabled:
            if _is_sqlite_conn(conn):
                sql = f"""
                SELECT signal_name, horizon_days, layer_name, weight, min_regime, enabled
                FROM {t}
                WHERE enabled = 1
                """
                cur.execute(sql)
            else:
                sql = f"""
                SELECT signal_name, horizon_days, layer_name, weight, min_regime, enabled
                FROM {t}
                WHERE enabled = TRUE
                """
                cur.execute(sql)
        else:
            cur.execute(
                f"SELECT signal_name, horizon_days, layer_name, weight, min_regime, enabled FROM {t}"
            )
        rows = cur.fetchall()
    finally:
        try:
            cur.close()
        except Exception:
            pass

    out: list[SignalLayerPolicyRow] = []
    for r in rows:
        if not r:
            continue
        out.append(
            SignalLayerPolicyRow(
                signal_name=str(r[0]),
                horizon_days=int(r[1]),
                layer_name=str(r[2]),
                weight=float(r[3]) if r[3] is not None else 0.0,
                min_regime=str(r[4] or "ANY"),
                enabled=bool(r[5]),
            )
        )
    return str(t), out


def score_sp500_from_signal_layer_policy(
    conn: Any,
    *,
    baseline_table: str = "sp500",
    baseline_symbol_col: str = "symbol",
    baseline_as_of_col: str | None = "date",
    baseline_as_of: str | date | datetime | None = None,
    as_of: str | date | datetime | None = None,
    policy_table: str = "intelligence.signal_layer_policy",
    score_performance_table: str = "score_performance_evaluation",
    quality_threshold: float = 0.0,
    confirm_threshold: float = 0.0,
    market_regime_table: str = "market_regime_daily",
    market_regime_date_col: str = "analysis_date",
) -> pd.DataFrame:
    """
    STEP 4: connect policy-driven layers to live per-stock scoring.

    Output columns (core):
    - symbol, date
    - market_regime_label, market_regime_score, market_policy_tag
    - layer_z_quality_365, layer_z_confirm_30, layer_z_micro_7
    - trade_state (DO_NOT_TRADE | WAIT | TRADE)
    - edge_score (ranking score; micro never overrides higher layers)
    """
    # Universe
    baseline = fetch_sp500_baseline(
        conn,
        table=baseline_table,
        symbol_col=baseline_symbol_col,
        as_of_col=baseline_as_of_col,
        as_of=baseline_as_of,
    )
    symbols = baseline[baseline_symbol_col].astype("string").dropna().tolist()
    if not symbols:
        return pd.DataFrame(
            columns=[
                "symbol",
                "date",
                "market_regime_label",
                "market_regime_score",
                "market_policy_tag",
                "layer_z_quality_365",
                "layer_z_confirm_30",
                "layer_z_micro_7",
                "trade_state",
                "edge_score",
            ]
        )

    # Use as_of (defaults to baseline_as_of when not provided).
    as_of_effective = as_of if as_of is not None else baseline_as_of

    # Market regime tag (POSITIVE/CAUTION) used for policy min_regime gating.
    regime_label: str | None = None
    regime_score: float | None = None
    try:
        mkt = fetch_market_regime_daily(
            conn,
            table=market_regime_table,
            analysis_date_col=market_regime_date_col,
            end=as_of_effective,
        )
        if not mkt.empty and market_regime_date_col in mkt.columns:
            m = mkt.copy()
            m[market_regime_date_col] = pd.to_datetime(
                m[market_regime_date_col], errors="coerce"
            )
            m = m.dropna(subset=[market_regime_date_col]).sort_values(
                market_regime_date_col
            )
            if not m.empty:
                last = m.iloc[-1]
                regime_label = str(last.get("regime_label")) if "regime_label" in m.columns else None
                try:
                    regime_score = (
                        float(last.get("regime_score"))
                        if "regime_score" in m.columns and last.get("regime_score") is not None
                        else None
                    )
                except Exception:
                    regime_score = None
    except Exception:
        pass
    market_tag = _market_tag_from_regime_row(regime_label, regime_score)

    # Policy rows
    _, policy_rows = _fetch_policy_rows(conn, table=policy_table, only_enabled=True)
    if not policy_rows:
        return pd.DataFrame(
            {
                "symbol": pd.Series(symbols, dtype="string"),
                "date": pd.Series([_as_iso_date(as_of_effective) if as_of_effective is not None else None] * len(symbols), dtype="string"),
                "market_regime_label": regime_label,
                "market_regime_score": regime_score,
                "market_policy_tag": market_tag,
                "layer_z_quality_365": 0.0,
                "layer_z_confirm_30": 0.0,
                "layer_z_micro_7": 0.0,
                "trade_state": "WAIT",
                "edge_score": 0.0,
            }
        )

    # Apply min_regime gating at the signal row level.
    active_policy: list[SignalLayerPolicyRow] = []
    for r in policy_rows:
        mr = str(r.min_regime or "ANY").upper()
        if mr == "ANY":
            active_policy.append(r)
        elif mr == str(market_tag).upper():
            active_policy.append(r)
    policy_rows = active_policy

    # Signal sources (wide tables in intelligence schema).
    from .selection import ScoreValueSource, fetch_latest_score_values_with_aliases

    sources = [
        ScoreValueSource("intelligence.nonuple_analysis", "symbol", "analysis_date"),
        ScoreValueSource("intelligence.nonuple_financial_metrics", "symbol", "analysis_date"),
        ScoreValueSource("intelligence.nonuple_technical_indicators", "symbol", "analysis_date"),
        ScoreValueSource("intelligence.mid_quarter_performance_scores", "symbol", "score_date"),
        ScoreValueSource("intelligence.options_flow_intelligence", "symbol", "analysis_date"),
        ScoreValueSource("intelligence.options_flow_summary", "symbol", "analysis_date"),
        ScoreValueSource("intelligence.trade_readiness_scores", "symbol", "score_date"),
        ScoreValueSource("intelligence.trade_phase_tracking", "symbol", "calculation_date"),
        ScoreValueSource("intelligence.ownership_intelligence", "symbol", "as_of"),
    ]

    score_names = sorted(set([r.signal_name for r in policy_rows]))
    sources2 = [
        ScoreValueSource(s.table.split(".")[-1], s.symbol_col, s.date_col) for s in sources
    ]
    try:
        raw = fetch_latest_score_values_with_aliases(
            conn, scores=score_names, sources=sources, as_of=as_of_effective
        )
    except Exception:
        raw = fetch_latest_score_values_with_aliases(
            conn, scores=score_names, sources=sources2, as_of=as_of_effective
        )
    # If schema-qualified tables are missing, discovery may return an empty frame.
    if raw.empty:
        raw = fetch_latest_score_values_with_aliases(
            conn, scores=score_names, sources=sources2, as_of=as_of_effective
        )
    if raw.empty:
        # No values available; return a safe empty-ish frame.
        out = pd.DataFrame({"symbol": pd.Series(symbols, dtype="string")})
        out["date"] = _as_iso_date(as_of_effective) if as_of_effective is not None else None
        out["market_regime_label"] = regime_label
        out["market_regime_score"] = regime_score
        out["market_policy_tag"] = market_tag
        out["layer_z_quality_365"] = 0.0
        out["layer_z_confirm_30"] = 0.0
        out["layer_z_micro_7"] = 0.0
        out["trade_state"] = "WAIT"
        out["edge_score"] = 0.0
        return out

    # Restrict to baseline symbols (SP500) explicitly.
    raw["symbol"] = raw["symbol"].astype("string")
    raw = raw[raw["symbol"].astype(str).isin(set(map(str, symbols)))].copy()
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "date",
                "market_regime_label",
                "market_regime_score",
                "market_policy_tag",
                "layer_z_quality_365",
                "layer_z_confirm_30",
                "layer_z_micro_7",
                "trade_state",
                "edge_score",
            ]
        )

    # Cross-sectional z-score per signal.
    zcols: dict[str, pd.Series] = {}
    for s in score_names:
        if s not in raw.columns:
            continue
        zcols[s] = _zscore(pd.to_numeric(raw[s], errors="coerce").astype("float64")).fillna(0.0)

    # Layer aggregation (Σ z_signal * weight).
    out = raw[["symbol", "date"]].copy()
    out["market_regime_label"] = regime_label
    out["market_regime_score"] = regime_score
    out["market_policy_tag"] = market_tag

    def _layer_sum(layer_name: str, horizon: int) -> pd.Series:
        xs = [r for r in policy_rows if r.layer_name == layer_name and int(r.horizon_days) == int(horizon)]
        if not xs:
            return pd.Series([0.0] * len(out), index=out.index, dtype="float64")
        acc = pd.Series([0.0] * len(out), index=out.index, dtype="float64")
        for r in xs:
            z = zcols.get(r.signal_name)
            if z is None:
                continue
            acc = acc + float(r.weight) * pd.to_numeric(z, errors="coerce").fillna(0.0)
        return acc.astype("float64")

    out["layer_z_quality_365"] = _layer_sum("quality_365", 365)
    out["layer_z_confirm_30"] = _layer_sum("confirm_30", 30)
    out["layer_z_micro_7"] = _layer_sum("micro_7", 7)

    # Hard hierarchy gates.
    q = pd.to_numeric(out["layer_z_quality_365"], errors="coerce").fillna(0.0)
    c = pd.to_numeric(out["layer_z_confirm_30"], errors="coerce").fillna(0.0)
    m = pd.to_numeric(out["layer_z_micro_7"], errors="coerce").fillna(0.0)

    trade_state = pd.Series(["WAIT"] * len(out), index=out.index, dtype="string")
    trade_state = trade_state.where(q >= float(quality_threshold), other="DO_NOT_TRADE")
    trade_state = trade_state.where(
        (trade_state == "DO_NOT_TRADE") | (c >= float(confirm_threshold)),
        other="WAIT",
    )
    # If both gates pass, we allow TRADE; micro influences timing/ranking only.
    trade_state = trade_state.where(
        (trade_state == "DO_NOT_TRADE") | (c < float(confirm_threshold)),
        other="TRADE",
    )
    out["trade_state"] = trade_state

    # Ranking score: micro never overrides higher layers (small weight).
    out["edge_score"] = (q + c + 0.25 * m).astype("float64")

    # Stable ordering for display: TRADE first, then WAIT, then DO_NOT_TRADE.
    prio = (
        out["trade_state"]
        .astype("string")
        .map({"TRADE": 0, "WAIT": 1, "DO_NOT_TRADE": 2})
        .fillna(9)
        .astype("int64")
    )
    out["_trade_prio"] = prio
    out = out.sort_values(["_trade_prio", "edge_score"], ascending=[True, False]).drop(
        columns=["_trade_prio"]
    )
    return out.reset_index(drop=True)


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
        # Degenerate distribution (e.g., singleton group): treat finite values as
        # "no cross-sectional signal" instead of propagating NaNs.
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        mask = np.isfinite(s.to_numpy())
        out.iloc[np.flatnonzero(mask)] = 0.0
        return out
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
    # NOTE: candidates are typically a single "as-of snapshot" (one row per symbol).
    # Grouping by date often creates singleton groups (stddev=0) which yields NaNs and
    # then SQL NULLs on persistence. Prefer cross-sectional scoring (optionally by regime).
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
        zq = _groupwise_zscore(out, "_raw_quality_365", keys)
        missing_q = ~np.isfinite(pd.to_numeric(zq, errors="coerce").to_numpy())
        out["layer_z_quality_365"] = pd.to_numeric(zq, errors="coerce").fillna(0.0)
        out["topdown_missing_quality_365"] = missing_q
    else:
        _LOG.info(
            "topdown: quality_365: no quality cols present (expected one of=%s)",
            list(ev.quality_cols),
        )
        out["layer_z_quality_365"] = 0.0
        out["topdown_missing_quality_365"] = True

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
            out[out_col] = 0.0
            out[f"topdown_missing_{out_col}"] = True
            return
        _LOG.info("topdown: %s: using cols=%s", out_col, cols)
        out[f"_raw_{out_col}"] = out[cols].mean(axis=1)
        z = _groupwise_zscore(out, f"_raw_{out_col}", keys)
        missing = ~np.isfinite(pd.to_numeric(z, errors="coerce").to_numpy())
        out[out_col] = pd.to_numeric(z, errors="coerce").fillna(0.0)
        out[f"topdown_missing_{out_col}"] = missing

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
    s = pd.to_numeric(out["layer_z_structural_63_126"], errors="coerce").astype(
        "float64"
    )
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
    out["edge_score_topdown"] = (
        (wq * q) + (ws * s) + (wc * c) + gate * ((wt * t) + (wm * micro_adj))
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
    # persistence (TopDownEvaluation only)
    persist_topdown: bool = False,
    topdown_run_table: str = "topdown_evaluation_run",
    topdown_result_table: str = "topdown_evaluation_result",
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
        # Optional persistence for the top-down strategy.
        if persist_topdown and isinstance(evaluation, TopDownEvaluation):
            run_id = persist_topdown_evaluation(
                c,
                scored,
                evaluation=evaluation,
                params={
                    "baseline_table": baseline_table,
                    "baseline_symbol_col": baseline_symbol_col,
                    "baseline_as_of_col": baseline_as_of_col,
                    "baseline_as_of": _as_iso_date(baseline_as_of)
                    if baseline_as_of is not None
                    else None,
                    "ohlcv_table": ohlcv_table,
                    "ohlcv_symbol_col": ohlcv_symbol_col,
                    "ohlcv_date_col": ohlcv_date_col,
                    "ohlcv_start": _as_iso_date(ohlcv_start)
                    if ohlcv_start is not None
                    else None,
                    "ohlcv_end": _as_iso_date(ohlcv_end)
                    if ohlcv_end is not None
                    else None,
                    "universe": universe,
                    "horizon_days": list(horizon_days)
                    if horizon_days is not None
                    else None,
                    "horizon_basis": horizon_basis,
                    "candidates_as_of": _as_iso_date(candidates_as_of)
                    if candidates_as_of is not None
                    else None,
                },
                run_table=topdown_run_table,
                result_table=topdown_result_table,
            )
            scored = scored.copy()
            scored["topdown_run_id"] = int(run_id)
            _LOG.info(
                "run_edge: persisted TopDownEvaluation results (run_id=%d table=%s)",
                int(run_id),
                topdown_result_table,
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
    evaluation: EdgeEvaluation | TopDownEvaluation | None = None,
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
        evaluation=evaluation
        if isinstance(evaluation, (EdgeEvaluation, TopDownEvaluation))
        else None,
        persist_topdown=bool(getattr(args, "persist_topdown", False)),
        topdown_run_table=str(
            getattr(args, "topdown_run_table", "topdown_evaluation_run")
        ),
        topdown_result_table=str(
            getattr(args, "topdown_result_table", "topdown_evaluation_result")
        ),
    )

    # Optional: filter to "qualified" symbols via the selection module.
    if bool(getattr(args, "select_qualified", False)):
        try:
            from .selection import pick_qualified_stocks

            select_as_of = getattr(args, "select_as_of", None) or getattr(
                args, "candidates_as_of", None
            )
            select_horizon = int(getattr(args, "select_horizon_days", 365) or 365)

            qualified = pick_qualified_stocks(
                conn,
                horizon_days=select_horizon,
                as_of=select_as_of,
                score_performance_table=str(getattr(args, "score_performance_table")),
                return_details=True,
            )
            if qualified is None or getattr(qualified, "empty", True):
                _LOG.warning(
                    "selection: --select-qualified requested but no actionable/qualified symbols found; skipping selection filter"
                )
            else:
                q = qualified.copy()
                # Avoid colliding with candidate `date` (which is OHLCV/feature as-of).
                if "date" in q.columns:
                    q = q.rename(columns={"date": "select_date"})
                keep_cols = [
                    c
                    for c in ("symbol", "select_date", "pass_count", "n_scores", "pass_rate")
                    if c in q.columns
                ]
                q = q[keep_cols].copy() if keep_cols else q
                out = out.merge(q, on="symbol", how="inner")
                _LOG.info(
                    "selection: filtered candidates to qualified symbols (rows=%d)",
                    len(out),
                )
        except ImportError as exc:
            # Selection is an optional add-on; if the selection module (or any of its
            # dependencies) isn't importable in a given environment, skip quietly.
            # Keep details available for debugging without spamming normal runs.
            _LOG.debug("selection: skipped (%s)", exc)
        except Exception as exc:
            _LOG.warning("selection: skipped (%s)", exc)

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

    def _pick_cli_columns(df: pd.DataFrame) -> list[str]:
        """
        Keep CLI output compact by default (avoid dumping hundreds of columns).

        Users can override with --print-all-columns.
        """
        preferred = [
            # identity
            "symbol",
            "universe",
            "date",
            # prices/liquidity (common)
            "close",
            "volume",
            "liquidity",
            # regime context (if joined)
            "regime",
            "regime_id",
            "risk_on_off",
            "vol_regime",
            "regime_score",
            "position_multiplier",
            "vix_level",
            "vix_zscore",
            "vix9d_over_vix",
            "vix_vix3m_ratio",
            "move_level",
            "move_zscore",
            "term_structure_state",
            "event_risk_flag",
            "joint_stress_flag",
            # score outputs
            "edge_score",
            "edge_score_topdown",
            "topdown_gate",
        ]
        cols = [c for c in preferred if c in df.columns]
        # Fallback: if the preferred set isn't present, show a small slice.
        if not cols:
            cols = list(df.columns[:12])
        return cols

    shown_to_print = shown
    if not getattr(args, "print_all_columns", False):
        cols = _pick_cli_columns(shown)
        shown_to_print = shown[cols].copy()

    # --- Print candidates (stable, human-friendly formatting)
    total_n = int(len(out))
    shown_n = int(len(shown_to_print))
    header = (
        f"Candidates (rows={total_n}, printed={shown_n}"
        + (", top=all" if not args.top else f", top={int(args.top)}")
        + "):"
    )
    print(header)
    print("-" * len(header))
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(
            shown_to_print.to_string(
                index=False,
                formatters=_make_cli_formatters(shown_to_print),
            )
        )

    # `fetch_score_name_groups()` returns a DataFrame, which cannot be used in a boolean
    # context (pandas raises: "The truth value of a DataFrame is ambiguous.").
    # Historically this CLI printed a small summary when groups were available; keep the
    # behavior but make it robust to both DataFrame and dict-like returns.
    if score_groups is not None:
        if isinstance(score_groups, pd.DataFrame):
            if not score_groups.empty:
                print("\nScore performance groups:")
                print("------------------------")

                # By default, do NOT print the full score_names lists (they can be huge).
                cols = [
                    c
                    for c in ["horizon_days", "regime_label", "n_scores"]
                    if c in score_groups.columns
                ]
                score_groups_to_print = score_groups

                if (
                    getattr(args, "print_score_names", False)
                    and "score_names" in score_groups.columns
                ):
                    max_names = int(getattr(args, "max_score_names", 25) or 25)
                    sg = score_groups.copy()
                    sg["score_names_preview"] = sg["score_names"].apply(
                        lambda xs: (
                            list(xs)[:max_names] if isinstance(xs, list) else xs
                        )
                    )
                    cols = cols + ["score_names_preview"]
                    score_groups_to_print = sg[cols]
                else:
                    score_groups_to_print = score_groups[cols] if cols else score_groups

                with pd.option_context(
                    "display.max_rows",
                    None,
                    "display.max_columns",
                    None,
                ):
                    print(score_groups_to_print.to_string(index=False))
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
        "--topdown",
        action="store_true",
        help="Use the top-down multi-horizon evaluation strategy (TopDownEvaluation).",
    )
    p.add_argument(
        "--policy",
        action="store_true",
        help="Use the signal_layer_policy-driven live scoring workflow (policy-weighted layers + hard gates).",
    )
    p.add_argument(
        "--policy-table",
        default="intelligence.signal_layer_policy",
        help="Policy table name. Default: intelligence.signal_layer_policy.",
    )
    p.add_argument(
        "--ensure-signal-layer-policy",
        action="store_true",
        help="Create the signal_layer_policy table (STEP 3) and exit.",
    )
    p.add_argument(
        "--seed-signal-layer-policy",
        action="store_true",
        help="Upsert the S&P 500 whitelist into signal_layer_policy (STEP 2/3).",
    )
    p.add_argument(
        "--validate-policy-thresholds",
        action="store_true",
        help="When seeding policy rows, disable any whitelist signals that fail the exact horizon thresholds (STEP 1).",
    )
    p.add_argument(
        "--policy-quality-threshold",
        type=float,
        default=0.0,
        help="Hard gate threshold for layer_z_quality_365 (STEP 4). Default: 0.0.",
    )
    p.add_argument(
        "--policy-confirm-threshold",
        type=float,
        default=0.0,
        help="Hard gate threshold for layer_z_confirm_30 (STEP 4). Default: 0.0.",
    )
    p.add_argument(
        "--persist-topdown",
        action="store_true",
        help="Persist TopDownEvaluation results into DB tables.",
    )
    p.add_argument(
        "--topdown-run-table",
        default="topdown_evaluation_run",
        help="Run-metadata table for persisted TopDownEvaluation outputs.",
    )
    p.add_argument(
        "--topdown-result-table",
        default="topdown_evaluation_result",
        help="Result table for persisted TopDownEvaluation outputs.",
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
        "--print-all-columns",
        action="store_true",
        help="Print all candidate columns (can be very wide). Default: print a compact subset.",
    )
    p.add_argument(
        "--print-score-names",
        action="store_true",
        help="Include a preview of score_names when printing score performance groups. Default: hidden.",
    )
    p.add_argument(
        "--max-score-names",
        type=int,
        default=25,
        help="Max score names to show when using --print-score-names. Default: 25.",
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
        "--select-qualified",
        action="store_true",
        help=(
            "Filter printed candidates to symbols qualified by actionable score_names "
            "(uses springedge.selection.pick_qualified_stocks)."
        ),
    )
    p.add_argument(
        "--select-as-of",
        default=None,
        help=(
            "As-of date for qualification selection (also used as candidates-as-of when "
            "--candidates-as-of is not provided)."
        ),
    )
    p.add_argument(
        "--select-horizon-days",
        type=int,
        default=365,
        help="Horizon (days) used for qualification selection. Default: 365.",
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

    # Compatibility: callers often think in selection terms ("select as-of").
    # If provided, treat it as the default candidates-as-of as well.
    if getattr(args, "candidates_as_of", None) is None and getattr(args, "select_as_of", None):
        args.candidates_as_of = args.select_as_of

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
            horizon_days = (7, 21, 30, 63, 126, 365) if args.topdown else (7, 21)

        evaluation: EdgeEvaluation | TopDownEvaluation | None = None
        if args.topdown:
            evaluation = TopDownEvaluation()

        try:
            return _run_and_print_edge(
                conn,
                args=args,
                baseline_as_of_col=baseline_as_of_col,
                horizon_days=horizon_days,
                evaluation=evaluation,
            )
        finally:
            try:
                conn.close()
            except Exception:
                pass

    with db_connection(args.db_url, env_var=args.db_env_var) as conn:
        # STEP 3/STEP 2: policy table management options.
        if bool(getattr(args, "ensure_signal_layer_policy", False)):
            tbl = ensure_signal_layer_policy_table(conn, table=str(args.policy_table))
            _LOG.info("signal_layer_policy: ensured table=%s", tbl)
            return 0
        if bool(getattr(args, "seed_signal_layer_policy", False)):
            rows = default_sp500_signal_layer_policy_rows()
            if bool(getattr(args, "validate_policy_thresholds", False)):
                # Evaluate exact thresholds and disable rows that fail.
                perf_cache: dict[int, pd.DataFrame] = {}
                enabled: list[SignalLayerPolicyRow] = []
                for r in rows:
                    h = int(r.horizon_days)
                    if h not in perf_cache:
                        try:
                            perf_cache[h] = evaluate_actionable_signals_by_horizon(
                                conn, horizon_days=h, table=str(args.score_performance_table)
                            )
                        except Exception as exc:
                            _LOG.warning(
                                "signal_layer_policy: threshold validation skipped for horizon=%s (%s)",
                                h,
                                exc,
                            )
                            perf_cache[h] = pd.DataFrame()
                    perf = perf_cache[h]
                    if perf.empty or "score_name" not in perf.columns:
                        enabled.append(r)
                        continue
                    hit = perf[perf["score_name"].astype(str) == str(r.signal_name)]
                    if hit.empty:
                        enabled.append(SignalLayerPolicyRow(**{**asdict(r), "enabled": False}))
                        _LOG.info(
                            "signal_layer_policy: disabled (no perf row) signal=%s horizon=%s",
                            r.signal_name,
                            r.horizon_days,
                        )
                        continue
                    ok = bool(hit["passed"].iloc[0]) if "passed" in hit.columns else True
                    enabled.append(SignalLayerPolicyRow(**{**asdict(r), "enabled": bool(ok)}))
                    if not ok:
                        reason = str(hit["reason"].iloc[0]) if "reason" in hit.columns else "fails thresholds"
                        _LOG.info(
                            "signal_layer_policy: disabled (fails thresholds) signal=%s horizon=%s reason=%s",
                            r.signal_name,
                            r.horizon_days,
                            reason,
                        )
                rows = enabled
            tbl = upsert_signal_layer_policy(conn, rows, table=str(args.policy_table))
            _LOG.info("signal_layer_policy: upserted rows=%d into table=%s", len(rows), tbl)
            return 0

        # STEP 4: policy-driven scoring workflow.
        if bool(getattr(args, "policy", False)):
            scored = score_sp500_from_signal_layer_policy(
                conn,
                baseline_table=args.baseline_table,
                baseline_symbol_col=args.baseline_symbol_col,
                baseline_as_of_col=baseline_as_of_col,  # type: ignore[arg-type]
                baseline_as_of=args.baseline_as_of,
                as_of=args.candidates_as_of or args.baseline_as_of,
                policy_table=str(args.policy_table),
                score_performance_table=str(args.score_performance_table),
                quality_threshold=float(args.policy_quality_threshold),
                confirm_threshold=float(args.policy_confirm_threshold),
                market_regime_table="market_regime_daily",
                market_regime_date_col="analysis_date",
            )

            shown = scored if not args.top else scored.head(int(args.top))
            if shown.empty:
                print("(no candidates)")
                return 0
            header = (
                f"Policy candidates (rows={len(scored)}, printed={len(shown)}"
                + (", top=all" if not args.top else f", top={int(args.top)}")
                + "):"
            )
            print(header)
            print("-" * len(header))
            cols = [
                c
                for c in [
                    "symbol",
                    "date",
                    "market_policy_tag",
                    "layer_z_quality_365",
                    "layer_z_confirm_30",
                    "layer_z_micro_7",
                    "trade_state",
                    "edge_score",
                ]
                if c in shown.columns
            ]
            shown_to_print = shown[cols] if cols else shown
            with pd.option_context("display.max_rows", None, "display.max_columns", None):
                print(
                    shown_to_print.to_string(
                        index=False,
                        formatters=_make_cli_formatters(shown_to_print),
                    )
                )
            return 0

        evaluation: EdgeEvaluation | TopDownEvaluation | None = (
            TopDownEvaluation() if args.topdown else None
        )
        return _run_and_print_edge(
            conn,
            args=args,
            baseline_as_of_col=baseline_as_of_col,
            horizon_days=horizon_days,
            evaluation=evaluation,
        )


if __name__ == "__main__":
    raise SystemExit(main())
