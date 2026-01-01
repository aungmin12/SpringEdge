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
    # Market regime join overwrites the label to the canonical market regime.
    assert out["regime"].tolist() == ["market_risk_on", "market_risk_on"]

    # Horizon-aware momentum columns exist (requested labels).
    for c in ["mom_ret_7d", "mom_ret_21d"]:
        assert c in out.columns

    # Indicator z-scores and overall score exist.
    assert "edge_score" in out.columns
    assert any(c.startswith("ind_z_") for c in out.columns)


def test_run_edge_end_to_end_sqlite_ohlcv_with_ticker_column_fallbacks():
    conn = sqlite3.connect(":memory:")
    conn.execute("ATTACH DATABASE ':memory:' AS core")
    conn.execute("create table sp500 (date text, symbol text)")
    # Common production variant: ticker column instead of symbol.
    conn.execute(
        "create table core.prices_daily (ticker text, date text, open real, high real, low real, close real, volume real)"
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
            "insert into core.prices_daily (ticker, date, open, high, low, close, volume) values (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

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
            "market_risk_off",
            -0.5,
            0.5,
            22.0,
            1.5,
            1.2,
            1.1,
            120.0,
            1.0,
            "stressed",
            1,
            1,
            "test",
            last_date,
        ),
    )

    out = run_edge(
        conn=conn,
        baseline_as_of="2024-02-01",
        horizon_days=(7, 21),
        horizon_basis="trading",
    )

    assert out["symbol"].tolist() == ["AAA", "BBB"]
    assert out["date"].notna().all()
    assert "edge_score" in out.columns
    assert out["regime"].tolist() == ["market_risk_off", "market_risk_off"]
