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
    from .layers import _validate_table_ref
except ImportError:
    _src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # .../src
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    from springedge.db import db_connection  # type: ignore
    from springedge.layers import _validate_table_ref  # type: ignore

_LOG = logging.getLogger(__name__)


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
    return any((f'relation "{c}" does not exist' in msg) or (f"no such table: {c}" in msg) for c in candidates)


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
        # If the table is missing, fall back to a common schema-qualified name. If both are
        # missing, return an empty result (this helper should be safe to call from broader
        # pipelines where the score_performance table may not exist yet).
        if _missing_table_error(exc, table_name=table):
            try:
                raw = _run_query(_table="intelligence.score_performance_evaluation")
            except Exception as exc2:
                _safe_rollback()
                if _missing_table_error(exc2, table_name="intelligence.score_performance_evaluation"):
                    return pd.DataFrame(columns=["horizon_days", "regime_label", "n_scores", "score_names"])
                raise
        else:
            raise

    if raw.empty:
        return pd.DataFrame(columns=["horizon_days", "regime_label", "n_scores", "score_names"])

    raw["score_name"] = raw["score_name"].astype("string")
    raw["regime_label"] = raw["regime_label"].astype("string")
    raw["horizon_days"] = pd.to_numeric(raw["horizon_days"], errors="coerce").astype("Int64")
    raw = raw.dropna(subset=["score_name", "horizon_days", "regime_label"]).copy()
    if raw.empty:
        return pd.DataFrame(columns=["horizon_days", "regime_label", "n_scores", "score_names"])

    grouped = (
        raw.groupby(["horizon_days", "regime_label"], dropna=False, sort=True)["score_name"]
        .apply(lambda s: sorted(set(s.dropna().astype(str).tolist())))
        .rename("score_names")
        .reset_index()
    )
    grouped["n_scores"] = grouped["score_names"].apply(len).astype("int64")
    grouped = grouped[["horizon_days", "regime_label", "n_scores", "score_names"]].sort_values(
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
    p.add_argument("--log-level", default="INFO", help="Logging level (e.g. DEBUG, INFO). Default: INFO.")
    p.add_argument("--demo", action="store_true", help="Run a self-contained sqlite demo (no external DB required).")
    p.add_argument("--db-url", default=None, help="Database URL (overrides env var if provided).")
    p.add_argument("--env-var", default="SPRINGEDGE_DB_URL", help="Env var containing DB URL. Default: SPRINGEDGE_DB_URL.")
    p.add_argument(
        "--table",
        default="score_performance_evaluation",
        help="Source table name. Default: score_performance_evaluation (also tries intelligence.score_performance_evaluation).",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.demo:
        import sqlite3

        _LOG.info("Running demo sqlite score_performance pipeline.")
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "create table score_performance_evaluation (score_name text, horizon_days integer, regime_label text)"
        )
        conn.executemany(
            "insert into score_performance_evaluation (score_name, horizon_days, regime_label) values (?, ?, ?)",
            [
                ("alpha", 21, "risk_on"),
                ("alpha", 21, "risk_on"),  # dup
                ("beta", 21, "risk_on"),
                ("alpha", 63, "risk_off"),
                ("gamma", 63, "risk_off"),
            ],
        )
        df = fetch_score_name_groups(conn, table=args.table)
        try:
            conn.close()
        except Exception:
            pass
    else:
        _LOG.info("Fetching score performance groups (table=%s).", args.table)
        with db_connection(args.db_url, env_var=args.env_var) as conn:
            df = fetch_score_name_groups(conn, table=args.table)

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


__all__ = ["fetch_score_name_groups", "main"]


if __name__ == "__main__":
    raise SystemExit(main())

