from __future__ import annotations

from typing import Any

import pandas as pd

from .layers import _validate_table_ref


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
        if _missing_table_error(exc, table_name=table):
            raw = _run_query(_table="intelligence.score_performance_evaluation")
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


__all__ = ["fetch_score_name_groups"]

