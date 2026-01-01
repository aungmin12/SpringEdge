from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Sequence

import pandas as pd

# This module is primarily intended to be run/imported as part of the `springedge`
# package. However, some users run `python3 edge.py ...` from inside `src/springedge/`,
# which imports this file without a known parent package. In that case, relative
# imports fail. We fall back to importing via the `springedge.*` package after
# ensuring `.../src` is on `sys.path`.
try:
    from .db import db_connection
    from .logging_utils import configure_logging
    from .layers import _validate_table_ref
except ImportError:
    _src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # .../src
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    from springedge.db import db_connection  # type: ignore
    from springedge.logging_utils import configure_logging  # type: ignore
    from springedge.layers import _validate_table_ref  # type: ignore

# If a user runs this file directly, `__name__` becomes "__main__" which is noisy.
_LOG = logging.getLogger(
    "springedge.score_performance" if __name__ == "__main__" else __name__
)

ACTIONABLE_MIN_ABS_SPEARMAN_IC = 0.10
ACTIONABLE_MIN_IC_IR = 1.5
# q5-q1 spread default threshold.
#
# Historically this was expressed as a decimal return (0.05 == 5%).
# Many users think in "percent points", so the default is now expressed as 5.0,
# while we still *compare* in decimal space internally (see
# `_normalize_q5_q1_threshold_to_decimals`).
ACTIONABLE_MIN_ABS_Q5_MINUS_Q1 = 5.0


def _rollback_quietly(conn: Any) -> None:
    try:
        if hasattr(conn, "rollback"):
            conn.rollback()
    except Exception:
        pass


def _missing_table_error(err: Exception, *, table_name: str) -> bool:
    """
    Best-effort detection of missing table errors across DB drivers.
    - Postgres: relation "x" does not exist
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


def _load_table_as_df(conn: Any, *, table: str) -> pd.DataFrame:
    """
    Load a table as a DataFrame (best-effort) via DB-API cursor.

    This avoids pandas.read_sql_query driver warnings for some DB-API connections.
    """
    t = _validate_table_ref(table, kind="table")
    sql = f"SELECT * FROM {t}"
    cur = conn.cursor()
    try:
        try:
            cur.execute(sql)
        except Exception:
            _rollback_quietly(conn)
            raise
        rows = cur.fetchall()
        cols = [d[0] for d in (cur.description or [])]
    finally:
        try:
            cur.close()
        except Exception:
            pass
    return pd.DataFrame.from_records(rows, columns=cols)


def _first_present(columns: list[str], candidates: Sequence[str]) -> str | None:
    cols = {str(c) for c in columns}
    for cand in candidates:
        if cand in cols:
            return cand
    return None


def _normalize_spread_units(x: pd.Series, *, unit: str = "auto") -> pd.Series:
    """
    Normalize q5-q1 spread units.

    Some tables store spreads in:
    - "raw" (already in the intended units, e.g. 5.0 means 5.0)
    - "percent_points" (e.g. 5.0 means 5%, i.e. 0.05 in decimals)

    `unit`:
    - "raw": no scaling
    - "percent_points": divide by 100
    - "auto": best-effort heuristic (backwards compatible with earlier behavior)
    """
    s = pd.to_numeric(x, errors="coerce").astype("float64")
    unit = str(unit or "auto").strip().lower()
    if unit == "raw":
        return s
    if unit in {"percent_points", "pct_points", "pp"}:
        return s / 100.0
    if unit == "auto":
        m = float(s.abs().dropna().median()) if not s.dropna().empty else 0.0
        # If typical magnitudes exceed ~1.5, it's very likely percent points.
        if m > 1.5:
            return s / 100.0
        return s
    raise ValueError(
        f"Unknown q5-q1 spread unit: {unit!r} (expected 'raw', 'percent_points', or 'auto')."
    )


def _normalize_q5_q1_threshold_to_decimals(x: float) -> float:
    """
    Normalize the q5-q1 absolute threshold to *decimal* return units.

    For backwards compatibility, callers may provide either:
    - a decimal return (e.g. 0.05 for 5%)
    - percent points (e.g. 5.0 for 5%)

    Heuristic:
    - values with |x| > 1.0 are treated as percent points and divided by 100
    - otherwise values are treated as already-decimal
    """
    v = float(x)
    return v / 100.0 if abs(v) > 1.0 else v


def _load_table_with_fallback(conn: Any, *, table: str) -> pd.DataFrame | None:
    """
    Load a score-performance table and apply a schema-qualified fallback.

    Behavior:
    - tries `table`
    - if missing:
      - when `table` is unqualified: tries `intelligence.<table>`
      - when `table` is qualified: tries the unqualified tail name
    - returns None if both are missing
    """
    try:
        return _load_table_as_df(conn, table=table)
    except Exception as exc:
        _rollback_quietly(conn)
        if not _missing_table_error(exc, table_name=table):
            raise

    fallback = (
        "intelligence." + str(table)
        if "." not in str(table)
        else str(table).split(".")[-1]
    )
    try:
        return _load_table_as_df(conn, table=fallback)
    except Exception as exc2:
        _rollback_quietly(conn)
        if _missing_table_error(exc2, table_name=fallback):
            return None
        raise


def _fetch_actionable_score_names_sql(
    conn: Any,
    *,
    table: str,
    horizon_days: int,
    min_abs_spearman_ic: float,
    min_ic_ir: float,
    min_abs_q5_minus_q1: float,
    require_all_regimes: bool,
    q5_q1_unit: str,
) -> list[str] | None:
    """
    Best-effort SQL pushdown for actionable filtering.

    Returns:
      - list[str] on success
      - None if we cannot/should not use SQL (unknown unit, unsupported SQL, etc.)

    Notes:
    - This intentionally avoids Postgres-only aggregates like bool_and/bool_or by using
      numeric aggregation over a CASE expression (portable across many DBs).
    - We only use this when q5-q1 units are explicit ("raw" or "percent_points").
      For "auto", we fall back to the pandas path because it relies on heuristics.
    """
    # Only attempt SQL pushdown on Postgres drivers where we can rely on %s placeholders.
    # This avoids side effects in sqlite tests (and avoids dialect differences).
    conn_mod = str(getattr(conn.__class__, "__module__", "") or "")
    if ("psycopg" not in conn_mod) and ("psycopg2" not in conn_mod):
        return None

    unit = str(q5_q1_unit or "auto").strip().lower()
    if unit == "auto":
        return None
    if unit == "raw":
        scale = 1.0
    elif unit in {"percent_points", "pct_points", "pp"}:
        # stored as percent points, normalize to decimals
        scale = 0.01
    else:
        return None

    t = _validate_table_ref(table, kind="table")

    # Portable aggregation:
    # - ALL regimes: MIN(passed_int) = 1
    # - ANY regime: MAX(passed_int) = 1
    agg = "MIN" if require_all_regimes else "MAX"

    sql = f"""
    WITH per_row AS (
      SELECT
        score_name,
        CASE
          WHEN ABS(spearman_ic) >= %s
           AND ic_ir >= %s
           AND ABS(q5_minus_q1 * %s) >= %s
          THEN 1 ELSE 0
        END AS passed_int
      FROM {t}
      WHERE horizon_days = %s
        AND score_name IS NOT NULL
    )
    SELECT score_name
    FROM per_row
    GROUP BY score_name
    HAVING {agg}(passed_int) = 1
    ORDER BY score_name
    """

    params = (
        float(min_abs_spearman_ic),
        float(min_ic_ir),
        float(scale),
        float(min_abs_q5_minus_q1),
        int(horizon_days),
    )

    cur = conn.cursor()
    try:
        try:
            cur.execute(sql, params)
        except Exception:
            # If the SQL attempt fails on Postgres, the transaction may now be aborted.
            # We must rollback before falling back to the pandas path to avoid
            # psycopg.errors.InFailedSqlTransaction on the next execute().
            _rollback_quietly(conn)
            return None
        rows = cur.fetchall()
    finally:
        try:
            cur.close()
        except Exception:
            pass

    out: list[str] = []
    for r in rows:
        if not r:
            continue
        out.append(str(r[0]))
    return out


def fetch_actionable_score_names(
    conn: Any,
    *,
    table: str = "score_performance_evaluation",
    horizon_days: int = 365,
    min_abs_spearman_ic: float = ACTIONABLE_MIN_ABS_SPEARMAN_IC,
    min_ic_ir: float = ACTIONABLE_MIN_IC_IR,
    min_abs_q5_minus_q1: float = ACTIONABLE_MIN_ABS_Q5_MINUS_Q1,
    require_all_regimes: bool = True,
    q5_q1_unit: str = "auto",
) -> list[str]:
    """
    Filter `score_name` values by ALL criteria:
    - A: |spearman_ic| >= min_abs_spearman_ic
    - B: ic_ir >= min_ic_ir
    - C: |q5-q1| >= min_abs_q5_minus_q1 (for the provided horizon_days; default 365)

    Table expectations (column names are best-effort / inferred):
    - score_name, horizon_days, (optional) regime_label
    - spearman_ic (or a close alias)
    - ic_ir (or a close alias)
    - either q5_minus_q1 (or alias) OR both q5 and q1 return columns

    If `require_all_regimes=True`, a score is actionable only if it passes the criteria
    for every row (typically each regime_label) at the given horizon_days.
    """
    min_abs_q5_minus_q1_dec = _normalize_q5_q1_threshold_to_decimals(
        min_abs_q5_minus_q1
    )

    # If the caller provides explicit spread units, we can push the whole filter
    # into SQL for speed and simplicity (and avoid loading the entire table).
    sql_res = _fetch_actionable_score_names_sql(
        conn,
        table=table,
        horizon_days=horizon_days,
        min_abs_spearman_ic=min_abs_spearman_ic,
        min_ic_ir=min_ic_ir,
        min_abs_q5_minus_q1=min_abs_q5_minus_q1_dec,
        require_all_regimes=require_all_regimes,
        q5_q1_unit=q5_q1_unit,
    )
    if sql_res is not None:
        return sorted(set(map(str, sql_res)))

    df = _load_table_with_fallback(conn, table=table)
    if df is None or df.empty:
        return []

    df.columns = [str(c) for c in df.columns]
    cols = list(df.columns)

    score_col = _first_present(cols, ("score_name", "score", "name"))
    horizon_col = _first_present(
        cols, ("horizon_days", "horizon", "horizon_day", "horizon_d")
    )
    regime_col = _first_present(
        cols, ("regime_label", "regime", "market_regime", "regime_name")
    )

    ic_col = _first_present(
        cols, ("spearman_ic", "rank_ic", "ic_spearman", "spearman_rank_ic", "ic")
    )
    ir_col = _first_present(cols, ("ic_ir", "icir", "ir", "information_ratio"))

    # Prefer an explicit spread column.
    spread_col = _first_present(
        cols,
        (
            "q5_minus_q1",
            "q5_q1",
            "q5_q1_spread",
            "q5q1",
            "q5_minus_q1_ret",
            "q5_q1_ret",
        ),
    )
    # Otherwise compute spread from quintile return columns.
    q5_col = _first_present(cols, ("q5", "q5_ret", "q5_return", "ret_q5", "q5_mean"))
    q1_col = _first_present(cols, ("q1", "q1_ret", "q1_return", "ret_q1", "q1_mean"))

    missing: list[str] = []
    if score_col is None:
        missing.append("score_name")
    if horizon_col is None:
        missing.append("horizon_days")
    if ic_col is None:
        missing.append("spearman_ic")
    if ir_col is None:
        missing.append("ic_ir")
    if spread_col is None and (q5_col is None or q1_col is None):
        missing.append("q5-q1 (spread column or q5 & q1 columns)")
    if missing:
        raise ValueError(
            f"score_performance table missing required columns: {', '.join(missing)}"
        )

    work = df.copy()
    work[score_col] = work[score_col].astype("string")
    work[horizon_col] = pd.to_numeric(work[horizon_col], errors="coerce").astype(
        "Int64"
    )
    work = work.dropna(subset=[score_col, horizon_col]).copy()
    work = work[work[horizon_col] == int(horizon_days)].copy()
    if work.empty:
        return []

    ic = pd.to_numeric(work[ic_col], errors="coerce").astype("float64")
    ir = pd.to_numeric(work[ir_col], errors="coerce").astype("float64")
    if spread_col is not None:
        spread = _normalize_spread_units(work[spread_col], unit=q5_q1_unit)
    else:
        spread = _normalize_spread_units(
            work[q5_col], unit=q5_q1_unit
        ) - _normalize_spread_units(work[q1_col], unit=q5_q1_unit)

    passed = (
        (ic.abs() >= float(min_abs_spearman_ic))
        & (ir >= float(min_ic_ir))
        & (spread.abs() >= float(min_abs_q5_minus_q1_dec))
    )
    work["_passed"] = passed.fillna(False)

    # If there is no regime column, treat as one group per score_name.
    gb = work.groupby(score_col, dropna=False, sort=True)["_passed"]
    if (not regime_col) or require_all_regimes:
        ok = gb.all()
    else:
        ok = gb.any()
    return sorted(ok[ok].index.astype(str).tolist())


def fetch_score_name_groups(
    conn: Any,
    *,
    table: str = "score_performance_evaluation",
) -> pd.DataFrame:
    """
    Fetch distinct `score_name` values grouped by `horizon_days` and `regime_label`.

    Returns a DataFrame with columns:
      - horizon_days (int)
      - regime_label (string)
      - n_scores (int)
      - score_names (list[str])

    Notes:
    - This function avoids SQL dialect-specific aggregation (array_agg/group_concat)
      by doing the grouping in pandas.
    - If `table` is missing, we also try the common schema-qualified name:
      `intelligence.score_performance_evaluation`.
    """

    def _safe_rollback() -> None:
        try:
            if hasattr(conn, "rollback"):
                conn.rollback()
        except Exception:
            pass

    def _run_query(*, _table: str) -> pd.DataFrame:
        t = _validate_table_ref(_table, kind="table")
        sql = f"""
        SELECT DISTINCT
          score_name,
          horizon_days,
          regime_label
        FROM {t}
        WHERE score_name IS NOT NULL
        """
        cur = conn.cursor()
        try:
            try:
                cur.execute(sql)
            except Exception:
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

    try:
        raw = _run_query(_table=table)
    except Exception as exc:
        _safe_rollback()
        # If the table is missing, fall back between common schema-qualified names.
        # If both are missing, return empty (safe for broader pipelines).
        if not _missing_table_error(exc, table_name=table):
            raise

        fallback = (
            "intelligence.score_performance_evaluation"
            if "." not in str(table)
            else str(table).split(".")[-1]
        )
        try:
            raw = _run_query(_table=fallback)
        except Exception as exc2:
            _safe_rollback()
            if _missing_table_error(exc2, table_name=fallback):
                return pd.DataFrame(
                    columns=["horizon_days", "regime_label", "n_scores", "score_names"]
                )
            raise

    if raw.empty:
        return pd.DataFrame(
            columns=["horizon_days", "regime_label", "n_scores", "score_names"]
        )

    raw["score_name"] = raw["score_name"].astype("string")
    raw["regime_label"] = raw["regime_label"].astype("string")
    raw["horizon_days"] = pd.to_numeric(raw["horizon_days"], errors="coerce").astype(
        "Int64"
    )
    raw = raw.dropna(subset=["score_name", "horizon_days", "regime_label"]).copy()
    if raw.empty:
        return pd.DataFrame(
            columns=["horizon_days", "regime_label", "n_scores", "score_names"]
        )

    grouped = (
        raw.groupby(["horizon_days", "regime_label"], dropna=False, sort=True)[
            "score_name"
        ]
        .apply(lambda s: sorted(set(s.dropna().astype(str).tolist())))
        .rename("score_names")
        .reset_index()
    )
    grouped["n_scores"] = grouped["score_names"].apply(len).astype("int64")
    grouped = grouped[
        ["horizon_days", "regime_label", "n_scores", "score_names"]
    ].sort_values(
        ["horizon_days", "regime_label"],
        kind="mergesort",
    )
    # Convert Int64 -> int where possible for cleaner display.
    grouped["horizon_days"] = grouped["horizon_days"].astype("int64")
    return grouped.reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI entrypoint: print distinct `score_name` values grouped by horizon/regime as JSON.

    Examples:
      - Demo (no DB required):
        python3 -m springedge.score_performance --demo

      - Real DB:
        export SPRINGEDGE_DB_URL="postgresql://USER:PASSWORD@HOST:5432/DBNAME"
        python3 -m springedge.score_performance --table score_performance_evaluation
    """
    p = argparse.ArgumentParser(
        prog="springedge.score_performance",
        description="Fetch distinct score_name values grouped by horizon_days and regime_label.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO). Default: INFO.",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Run a self-contained sqlite demo (no external DB required).",
    )
    p.add_argument(
        "--db-url", default=None, help="Database URL (overrides env var if provided)."
    )
    p.add_argument(
        "--env-var",
        default="SPRINGEDGE_DB_URL",
        help="Env var containing DB URL. Default: SPRINGEDGE_DB_URL.",
    )
    p.add_argument(
        "--table",
        default="intelligence.score_performance_evaluation",
        help=(
            "Source table name. Default: intelligence.score_performance_evaluation "
            "(if missing, also tries score_performance_evaluation)."
        ),
    )
    p.add_argument(
        "--actionable",
        action="store_true",
        help="Print only actionable score_name values (filters by |spearman_ic|, ic_ir, and |q5-q1| at a horizon).",
    )
    p.add_argument(
        "--horizon-days",
        type=int,
        default=365,
        help="Horizon (days) used for actionable filtering. Default: 365.",
    )
    # Intentionally hard-coded actionable thresholds to keep CLI simple.
    # If you need configurability, call `fetch_actionable_score_names(...)` directly.
    regime_group = p.add_mutually_exclusive_group()
    regime_group.add_argument(
        "--all-regimes",
        dest="regime_mode",
        action="store_const",
        const="all",
        help=(
            "Require a score to pass the actionable thresholds in ALL regime rows "
            "(stricter; previously the default)."
        ),
    )
    regime_group.add_argument(
        "--any-regime",
        dest="regime_mode",
        action="store_const",
        const="any",
        help=(
            "A score is actionable if it passes in ANY regime row. "
            "This is now the default (matches SQL like `SELECT DISTINCT score_name ... WHERE ...`)."
        ),
    )
    p.set_defaults(regime_mode="any")
    args = p.parse_args(list(argv) if argv is not None else None)

    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    configure_logging(level=level)

    if args.demo:
        import sqlite3

        _LOG.info("Running demo sqlite score_performance pipeline.")
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "create table score_performance_evaluation (score_name text, horizon_days integer, regime_label text, spearman_ic real, ic_ir real, q5_minus_q1 real)"
        )
        conn.executemany(
            "insert into score_performance_evaluation (score_name, horizon_days, regime_label, spearman_ic, ic_ir, q5_minus_q1) values (?, ?, ?, ?, ?, ?)",
            [
                # 365d + two regimes
                ("alpha", 365, "risk_on", 0.12, 1.6, 0.06),
                ("alpha", 365, "risk_off", 0.11, 1.7, 0.055),
                # fails A
                ("beta", 365, "risk_on", 0.05, 3.0, 0.20),
                # fails B
                ("gamma", 365, "risk_on", 0.20, 1.0, 0.20),
                # fails C
                ("delta", 365, "risk_on", 0.20, 2.0, 0.01),
            ],
        )
        if args.actionable:
            scores = fetch_actionable_score_names(
                conn,
                table=args.table,
                horizon_days=args.horizon_days,
                require_all_regimes=not args.any_regime,
            )
            df = pd.DataFrame({"score_name": scores})
        else:
            df = fetch_score_name_groups(conn, table=args.table)
        try:
            conn.close()
        except Exception:
            pass
    else:
        if args.actionable:
            _LOG.info(
                "Filtering actionable score names (table=%s horizon_days=%s regime_mode=%s).",
                args.table,
                args.horizon_days,
                args.regime_mode,
            )
        else:
            _LOG.info("Fetching score performance groups (table=%s).", args.table)
        with db_connection(args.db_url, env_var=args.env_var) as conn:
            if args.actionable:
                scores = fetch_actionable_score_names(
                    conn,
                    table=args.table,
                    horizon_days=args.horizon_days,
                    require_all_regimes=(args.regime_mode == "all"),
                )
                df = pd.DataFrame({"score_name": scores})
            else:
                df = fetch_score_name_groups(conn, table=args.table)

    if args.actionable:
        payload = (
            df["score_name"].astype("string").dropna().astype(str).tolist()
            if "score_name" in df.columns
            else []
        )
        print(json.dumps(payload, indent=2, sort_keys=False))
        _LOG.info("Done (actionable_scores=%d).", len(payload))
    else:
        payload = [
            {
                "horizon_days": int(row.horizon_days),
                "regime_label": str(row.regime_label),
                "n_scores": int(row.n_scores),
                "score_names": list(row.score_names),
            }
            for row in df.itertuples(index=False)
        ]
        print(json.dumps(payload, indent=2, sort_keys=False))
        _LOG.info("Done (groups=%d).", len(payload))
    return 0


__all__ = ["fetch_actionable_score_names", "fetch_score_name_groups", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
