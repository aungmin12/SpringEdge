from __future__ import annotations

import argparse
import logging
from typing import Sequence

import pandas as pd

from springedge import db_connection
from springedge.regime import fetch_market_regime_last_n_days, market_regime_counts_and_trend
from springedge.logging_utils import configure_logging


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="market_regime_report",
        description="Fetch last N days from market_regime_daily and summarize dominant regime + recent trend.",
    )
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--db-url", default=None, help="Database URL (overrides env var if provided).")
    p.add_argument("--env-var", default="SPRINGEDGE_DB_URL", help="Env var containing DB URL. Default: SPRINGEDGE_DB_URL.")
    p.add_argument("--table", default="market_regime_daily")
    p.add_argument("--analysis-date-col", default="analysis_date")
    p.add_argument("--lookback-days", type=int, default=90)
    p.add_argument("--trend-days", type=int, default=10)
    p.add_argument("--as-of", default=None, help="As-of date (YYYY-MM-DD). Default: latest available analysis_date in table.")
    args = p.parse_args(list(argv) if argv is not None else None)

    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    configure_logging(level=level)

    with db_connection(args.db_url, env_var=args.env_var) as conn:
        df = fetch_market_regime_last_n_days(
            conn,
            lookback_days=args.lookback_days,
            as_of=args.as_of,
            table=args.table,
            analysis_date_col=args.analysis_date_col,
        )
        if df.empty:
            print("No rows returned from market_regime_daily for the requested window.")
            return 2

        counts, summary = market_regime_counts_and_trend(
            conn,
            lookback_days=args.lookback_days,
            trend_days=args.trend_days,
            as_of=args.as_of,
            table=args.table,
            analysis_date_col=args.analysis_date_col,
        )

    # Reproduce the SQL the user asked for (but computed from the fetched window):
    # SELECT regime_label, COUNT(*) AS cnt FROM market_regime_daily ... GROUP BY regime_label ORDER BY cnt DESC
    sql_counts = (
        df.groupby("regime_label", dropna=True)
        .size()
        .rename("cnt")
        .reset_index()
        .sort_values(["cnt", "regime_label"], ascending=[False, True])
        .reset_index(drop=True)
    )

    print("\nCounts by regime_label (last %d days, as-of %s):" % (args.lookback_days, summary.as_of))
    print(sql_counts.to_string(index=False))

    print("\nDominant market condition (last %d days):" % summary.lookback_days)
    print(
        "dominant_regime=%s dominant_days=%d total_days=%d share=%.3f"
        % (
            summary.dominant_regime,
            summary.dominant_days,
            summary.total_days,
            (summary.dominant_days / float(summary.total_days)) if summary.total_days else 0.0,
        )
    )

    print("\nTrend (last %d days):" % args.trend_days)
    print(
        "latest_regime=%s streak_days=%d recent_share_latest=%.3f recent_score_slope=%s"
        % (
            summary.latest_regime,
            summary.latest_streak_days,
            summary.recent_share_latest,
            ("%.6f" % summary.recent_score_slope) if isinstance(summary.recent_score_slope, float) else "None",
        )
    )

    # Optional: show last `trend_days` of labels (helps sanity-check “trend”).
    tail = df.copy()
    tail[args.analysis_date_col] = pd.to_datetime(tail[args.analysis_date_col], errors="coerce")
    tail = tail.dropna(subset=[args.analysis_date_col]).sort_values(args.analysis_date_col)
    recent = tail[[args.analysis_date_col, "regime_label"]].tail(int(args.trend_days))
    recent = recent.rename(columns={args.analysis_date_col: "analysis_date"})
    print("\nLast %d rows (date, regime_label):" % args.trend_days)
    print(recent.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

