"""
Stock selection utilities.

This module bridges the gap between:
- score performance *metadata* (e.g. intelligence.score_performance_evaluation)
and
- per-symbol score *values* (e.g. intelligence.nonuple_* tables)

It answers: “given actionable score_names, which symbols are currently qualified?”
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd

from .layers import _as_iso_date, _validate_table_ref
from .score_performance import fetch_actionable_score_specs


@dataclass(frozen=True)
class ScoreValueSource:
    """
    A source table that contains per-symbol score values.

    Example:
      intelligence.nonuple_technical_indicators has (symbol, analysis_date, rsi, ...)
    """

    table: str
    symbol_col: str = "symbol"
    date_col: str = "analysis_date"


def _is_sqlite_conn(conn: Any) -> bool:
    mod = str(getattr(conn.__class__, "__module__", "") or "")
    return "sqlite3" in mod


def _conn_placeholder(conn: Any) -> str:
    """
    Best-effort DB-API placeholder style:
    - sqlite3: '?'
    - psycopg/psycopg2: '%s'
    """
    return "?" if _is_sqlite_conn(conn) else "%s"


def _safe_rollback(conn: Any) -> None:
    try:
        if hasattr(conn, "rollback"):
            conn.rollback()
    except Exception:
        pass


def fetch_table_columns(conn: Any, *, table: str) -> list[str]:
    """
    Best-effort column discovery via a LIMIT 0 query.
    """
    t = _validate_table_ref(table, kind="table")
    cur = conn.cursor()
    try:
        try:
            cur.execute(f"SELECT * FROM {t} WHERE 1=0")
        except Exception as exc:
            _safe_rollback(conn)
            # Best-effort: if the table is missing, treat as "no columns" so
            # discovery can fall back to other sources.
            msg = str(exc).lower()
            tn = str(table).lower()
            candidates = {tn}
            if "." in tn:
                candidates.add(tn.split(".")[-1])
            if any(
                (f'relation "{c}" does not exist' in msg) or (f"no such table: {c}" in msg)
                for c in candidates
            ):
                return []
            raise
        cols = [d[0] for d in (cur.description or [])]
    finally:
        try:
            cur.close()
        except Exception:
            pass
    return [str(c) for c in cols]


def discover_score_sources(
    conn: Any,
    *,
    score_names: Iterable[str],
    sources: Iterable[ScoreValueSource],
) -> dict[str, ScoreValueSource]:
    """
    Map score_name -> first source table that contains a column with that name.
    """
    wanted = [str(s) for s in score_names if str(s).strip() != ""]
    if not wanted:
        return {}

    # Preload each table's column set once.
    table_cols: dict[str, set[str]] = {}
    for src in sources:
        cols = fetch_table_columns(conn, table=src.table)
        table_cols[src.table] = set(map(str, cols))

    out: dict[str, ScoreValueSource] = {}
    for s in wanted:
        for src in sources:
            if s in table_cols.get(src.table, set()):
                out[s] = src
                break
    return out


def _candidate_column_names(score_name: str) -> list[str]:
    """
    Best-effort mapping from a logical score_name to possible physical column names.

    Production score catalogs often use dotted names (e.g. "nonuple.growth"), while
    DB tables are typically wide and use identifier-safe columns (e.g. nonuple_growth).

    We attempt a few common normalizations:
    - exact name (for already-safe names)
    - replace '.' with '_' (nonuple.growth -> nonuple_growth)
    - last segment (nonuple.growth -> growth)
    """
    s = str(score_name or "").strip()
    if not s:
        return []
    cands: list[str] = [s]
    if "." in s:
        cands.append(s.replace(".", "_"))
        tail = s.split(".")[-1].strip()
        if tail:
            cands.append(tail)
    # stable de-dupe
    out: list[str] = []
    seen: set[str] = set()
    for c in cands:
        c = str(c).strip()
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def discover_score_sources_with_aliases(
    conn: Any,
    *,
    score_names: Iterable[str],
    sources: Iterable[ScoreValueSource],
) -> dict[str, tuple[ScoreValueSource, str]]:
    """
    Map requested score_name -> (source_table, actual_column_name).

    Unlike `discover_score_sources`, this supports dotted score names by trying
    `_candidate_column_names(score_name)` against each source's column set.
    """
    wanted = [str(s) for s in score_names if str(s).strip() != ""]
    if not wanted:
        return {}

    # Preload each table's column set once.
    table_cols: dict[str, set[str]] = {}
    for src in sources:
        cols = fetch_table_columns(conn, table=src.table)
        table_cols[src.table] = set(map(str, cols))

    out: dict[str, tuple[ScoreValueSource, str]] = {}
    for requested in wanted:
        for cand in _candidate_column_names(requested):
            for src in sources:
                if cand in table_cols.get(src.table, set()):
                    out[requested] = (src, cand)
                    break
            if requested in out:
                break
    return out


def fetch_latest_score_values(
    conn: Any,
    *,
    scores: list[str],
    sources: list[ScoreValueSource],
    as_of: str | date | datetime | None = None,
    start: str | date | datetime | None = None,
) -> pd.DataFrame:
    """
    Fetch latest per-symbol values for the requested score columns.

    Returns a DataFrame with:
      - symbol
      - date
      - one column per score_name present in the sources
    """
    if not scores:
        return pd.DataFrame(columns=["symbol", "date"])

    # Map score -> source table.
    mapping = discover_score_sources(conn, score_names=scores, sources=sources)
    if not mapping:
        return pd.DataFrame(columns=["symbol", "date"])

    # Group requested score columns per source table to minimize queries.
    by_table: dict[str, list[str]] = {}
    for s in scores:
        src = mapping.get(s)
        if src is None:
            continue
        by_table.setdefault(src.table, []).append(str(s))

    ph = _conn_placeholder(conn)
    as_of_iso = _as_iso_date(as_of) if as_of is not None else None
    start_iso = _as_iso_date(start) if start is not None else None

    frames: list[pd.DataFrame] = []
    for src in sources:
        cols = by_table.get(src.table)
        if not cols:
            continue
        t = _validate_table_ref(src.table, kind="table")
        sym_c = _validate_table_ref(src.symbol_col, kind="column")  # ident validation
        dt_c = _validate_table_ref(src.date_col, kind="column")

        # NOTE: We embed identifiers (validated) and only parameterize values.
        where: list[str] = []
        params: list[Any] = []
        if start_iso is not None:
            where.append(f"{dt_c} >= {ph}")
            params.append(start_iso)
        if as_of_iso is not None:
            where.append(f"{dt_c} <= {ph}")
            params.append(as_of_iso)
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        # Pull potentially multiple rows per symbol (then reduce in pandas).
        # This is portable across sqlite/postgres without relying on dialect-specific DISTINCT ON.
        cols_sql = ", ".join([sym_c, dt_c] + [_validate_table_ref(c, kind="column") for c in cols])
        sql = f"SELECT {cols_sql} FROM {t} {where_sql}"

        cur = conn.cursor()
        try:
            try:
                cur.execute(sql, tuple(params))
            except Exception:
                _safe_rollback(conn)
                raise
            rows = cur.fetchall()
            out_cols = [d[0] for d in (cur.description or [])]
        finally:
            try:
                cur.close()
            except Exception:
                pass

        df = pd.DataFrame.from_records(rows, columns=out_cols)
        if df.empty:
            continue

        # Normalize canonical names for merging.
        df = df.rename(columns={src.symbol_col: "symbol", src.date_col: "date"})
        df["symbol"] = df["symbol"].astype("string")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

        # Latest row per symbol.
        df = df.dropna(subset=["symbol", "date"]).copy()
        if df.empty:
            continue
        idx = df.groupby("symbol", sort=False)["date"].idxmax()
        df = df.loc[idx].reset_index(drop=True)
        frames.append(df[["symbol", "date", *cols]])

    if not frames:
        return pd.DataFrame(columns=["symbol", "date"])

    # Outer-merge all per-table frames on symbol/date.
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["symbol", "date"], how="outer")
    return out


def fetch_latest_score_values_with_aliases(
    conn: Any,
    *,
    scores: list[str],
    sources: list[ScoreValueSource],
    as_of: str | date | datetime | None = None,
    start: str | date | datetime | None = None,
) -> pd.DataFrame:
    """
    Like `fetch_latest_score_values`, but supports dotted score names.

    - Selects physical columns using `discover_score_sources_with_aliases`
    - Renames selected columns back to the requested score_name(s) in the output
    """
    if not scores:
        return pd.DataFrame(columns=["symbol", "date"])

    mapping = discover_score_sources_with_aliases(
        conn, score_names=scores, sources=sources
    )
    if not mapping:
        return pd.DataFrame(columns=["symbol", "date"])

    # Group requested score columns per source table to minimize queries.
    by_table: dict[str, list[tuple[str, str]]] = {}
    for requested in scores:
        m = mapping.get(requested)
        if m is None:
            continue
        src, actual = m
        by_table.setdefault(src.table, []).append((str(requested), str(actual)))

    ph = _conn_placeholder(conn)
    as_of_iso = _as_iso_date(as_of) if as_of is not None else None
    start_iso = _as_iso_date(start) if start is not None else None

    frames: list[pd.DataFrame] = []
    for src in sources:
        pairs = by_table.get(src.table)
        if not pairs:
            continue
        requested_cols = [p[0] for p in pairs]
        actual_cols = [p[1] for p in pairs]

        t = _validate_table_ref(src.table, kind="table")
        sym_c = _validate_table_ref(src.symbol_col, kind="column")  # ident validation
        dt_c = _validate_table_ref(src.date_col, kind="column")

        where: list[str] = []
        params: list[Any] = []
        if start_iso is not None:
            where.append(f"{dt_c} >= {ph}")
            params.append(start_iso)
        if as_of_iso is not None:
            where.append(f"{dt_c} <= {ph}")
            params.append(as_of_iso)
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        cols_sql = ", ".join(
            [sym_c, dt_c]
            + [_validate_table_ref(c, kind="column") for c in actual_cols]
        )
        sql = f"SELECT {cols_sql} FROM {t} {where_sql}"

        cur = conn.cursor()
        try:
            try:
                cur.execute(sql, tuple(params))
            except Exception:
                _safe_rollback(conn)
                raise
            rows = cur.fetchall()
            out_cols = [d[0] for d in (cur.description or [])]
        finally:
            try:
                cur.close()
            except Exception:
                pass

        df = pd.DataFrame.from_records(rows, columns=out_cols)
        if df.empty:
            continue

        df = df.rename(columns={src.symbol_col: "symbol", src.date_col: "date"})
        df["symbol"] = df["symbol"].astype("string")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        rename_map: dict[str, str] = {}
        for requested, actual in pairs:
            if actual in df.columns:
                df[actual] = pd.to_numeric(df[actual], errors="coerce").astype("float64")
                rename_map[actual] = requested
        if rename_map:
            df = df.rename(columns=rename_map)

        df = df.dropna(subset=["symbol", "date"]).copy()
        if df.empty:
            continue
        idx = df.groupby("symbol", sort=False)["date"].idxmax()
        df = df.loc[idx].reset_index(drop=True)
        keep = ["symbol", "date"] + [c for c in requested_cols if c in df.columns]
        frames.append(df[keep])

    if not frames:
        return pd.DataFrame(columns=["symbol", "date"])

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["symbol", "date"], how="outer")
    return out


def pick_qualified_stocks(
    conn: Any,
    *,
    horizon_days: int = 365,
    as_of: str | date | datetime | None = None,
    start: str | date | datetime | None = None,
    regime_label: str | None = None,
    # thresholds for actionable selection
    min_abs_spearman_ic: float | None = None,
    min_ic_ir: float | None = None,
    min_abs_q5_minus_q1: float | None = None,
    require_all_regimes: bool = True,
    q5_q1_unit: str = "auto",
    require_monotonic_quantiles: bool | None = None,
    min_sample_size: int | None = None,
    # stock qualification policy
    pass_quantile: float = 0.8,
    min_pass_fraction: float = 0.5,
    # sources
    score_performance_table: str = "score_performance_evaluation",
    sources: list[ScoreValueSource] | None = None,
    return_details: bool = True,
) -> pd.DataFrame:
    """
    Pick qualified symbols based on actionable score_names and current score values.

    Mechanics:
    - Determine actionable score_names at the given horizon using score_performance stats.
    - Infer direction (higher is better vs lower is better) from IC sign.
    - Fetch latest per-symbol values for those score_names from the provided sources.
    - For each score, compute cross-sectional percentile ranks.
      - direction=+1 passes if pct_rank >= pass_quantile
      - direction=-1 passes if pct_rank <= (1-pass_quantile)
    - A symbol is "qualified" if pass_rate >= min_pass_fraction.

    Returns:
    - If return_details=True: columns include pass_count, n_scores, pass_rate and per-score pass flags.
    - Else: a single-column DataFrame with symbols.
    """
    # 1) Build actionable specs (score_name + direction).
    kwargs: dict[str, Any] = dict(
        table=score_performance_table,
        horizon_days=int(horizon_days),
        require_all_regimes=bool(require_all_regimes),
        q5_q1_unit=str(q5_q1_unit),
        regime_label=regime_label,
        min_sample_size=min_sample_size,
        require_monotonic_quantiles=require_monotonic_quantiles,
        start_date=_as_iso_date(start) if start is not None else None,
        end_date=_as_iso_date(as_of) if as_of is not None else None,
    )
    if min_abs_spearman_ic is not None:
        kwargs["min_abs_spearman_ic"] = float(min_abs_spearman_ic)
    if min_ic_ir is not None:
        kwargs["min_ic_ir"] = float(min_ic_ir)
    if min_abs_q5_minus_q1 is not None:
        kwargs["min_abs_q5_minus_q1"] = float(min_abs_q5_minus_q1)

    specs = fetch_actionable_score_specs(conn, **kwargs)
    if not specs:
        return pd.DataFrame(columns=["symbol"] if not return_details else ["symbol", "reason"])

    # 2) Default sources: intelligence schema tables commonly used in this repo.
    srcs = sources or [
        ScoreValueSource("intelligence.nonuple_technical_indicators", "symbol", "analysis_date"),
        ScoreValueSource("intelligence.nonuple_financial_metrics", "symbol", "analysis_date"),
        ScoreValueSource("intelligence.nonuple_analysis", "symbol", "analysis_date"),
        ScoreValueSource("intelligence.mid_quarter_performance_scores", "symbol", "score_date"),
        ScoreValueSource("intelligence.options_flow_intelligence", "symbol", "analysis_date"),
        ScoreValueSource("intelligence.options_flow_summary", "symbol", "analysis_date"),
        ScoreValueSource("intelligence.trade_readiness_scores", "symbol", "score_date"),
        ScoreValueSource("intelligence.trade_phase_tracking", "symbol", "calculation_date"),
        ScoreValueSource("intelligence.ray_dalio_empire_analysis", "symbol", "analysis_date"),
        ScoreValueSource("intelligence.ownership_intelligence", "symbol", "as_of"),
    ]

    score_names = [s.score_name for s in specs]
    values = fetch_latest_score_values(conn, scores=score_names, sources=srcs, as_of=as_of, start=start)
    if values.empty:
        return pd.DataFrame(columns=["symbol"] if not return_details else ["symbol", "reason"])

    # 3) Compute pass flags per score.
    df = values.copy()
    df["symbol"] = df["symbol"].astype("string")
    df = df.dropna(subset=["symbol"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["symbol"] if not return_details else ["symbol", "reason"])

    # Percentile rank per score (stable, avoids qcut duplicate-edge issues).
    pass_cols: list[str] = []
    hi = float(pass_quantile)
    lo = float(1.0 - pass_quantile)
    for spec in specs:
        c = spec.score_name
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").astype("float64")
        pct = s.rank(pct=True, method="average")
        passed = pct >= hi if int(spec.direction) >= 0 else pct <= lo
        out_col = f"pass__{c}"
        df[out_col] = passed.fillna(False)
        pass_cols.append(out_col)

    if not pass_cols:
        return pd.DataFrame(columns=["symbol"] if not return_details else ["symbol", "reason"])

    df["pass_count"] = df[pass_cols].sum(axis=1).astype("int64")
    df["n_scores"] = int(len(pass_cols))
    df["pass_rate"] = (df["pass_count"] / df["n_scores"]).astype("float64")
    qualified = df[df["pass_rate"] >= float(min_pass_fraction)].copy()
    qualified = qualified.sort_values(["pass_rate", "pass_count"], ascending=False)

    if not return_details:
        return qualified[["symbol"]].reset_index(drop=True)

    # Keep output compact: symbol + summary + per-score pass flags.
    out_cols = ["symbol", "date", "pass_count", "n_scores", "pass_rate"] + pass_cols
    out_cols = [c for c in out_cols if c in qualified.columns]
    return qualified[out_cols].reset_index(drop=True)


__all__ = [
    "ScoreValueSource",
    "fetch_table_columns",
    "discover_score_sources",
    "discover_score_sources_with_aliases",
    "fetch_latest_score_values",
    "fetch_latest_score_values_with_aliases",
    "pick_qualified_stocks",
]

