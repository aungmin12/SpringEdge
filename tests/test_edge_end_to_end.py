import sqlite3

import numpy as np
import pandas as pd

from springedge import run_edge


def test_run_edge_end_to_end_sqlite_baseline_to_scored_candidates():
    conn = sqlite3.connect(":memory:")
    conn.execute("ATTACH DATABASE ':memory:' AS core")
    conn.execute("create table sp500 (date text, symbol text)")
    conn.execute(
        "create table core.prices_daily (symbol text, date text, open real, high real, low real, close real, volume real)"
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

    # Simple synthetic OHLCV for both symbols across enough bars to create momentum/risk features.
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

    out = run_edge(
        conn=conn,
        baseline_as_of="2024-02-01",
        horizon_days=(7, 21),
        horizon_basis="trading",
    )

    # One candidate row per baseline symbol.
    assert out["symbol"].tolist() == ["AAA", "BBB"]
    assert out["date"].notna().all()

    # Regime features exist.
    for c in ["regime", "regime_id", "risk_on_off", "vol_regime"]:
        assert c in out.columns

    # Horizon-aware momentum columns exist (requested labels).
    for c in ["mom_ret_7d", "mom_ret_21d"]:
        assert c in out.columns

    # Indicator z-scores and overall score exist.
    assert "edge_score" in out.columns
    assert any(c.startswith("ind_z_") for c in out.columns)

