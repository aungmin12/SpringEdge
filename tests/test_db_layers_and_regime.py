import sqlite3

import pandas as pd

from springedge.layers import fetch_sp500_baseline
from springedge.regime import (
    fetch_market_regime_last_n_days,
    fetch_market_regime_daily,
    fetch_regime_daily,
    market_regime_counts_and_trend,
    quarterly_regime_profile,
    summarize_market_regime,
)


def test_fetch_sp500_baseline_uses_latest_snapshot_by_default():
    conn = sqlite3.connect(":memory:")
    conn.execute("create table sp500 (date text, symbol text)")
    conn.executemany(
        "insert into sp500 (date, symbol) values (?, ?)",
        [
            ("2024-01-02", "AAPL"),
            ("2024-01-02", "MSFT"),
            ("2024-02-01", "AAPL"),
            ("2024-02-01", "NVDA"),
        ],
    )

    out = fetch_sp500_baseline(conn)
    assert out["symbol"].tolist() == ["AAPL", "NVDA"]


def test_fetch_regime_daily_and_quarterly_profile_highlights_dominant_last_quarter():
    conn = sqlite3.connect(":memory:")
    conn.execute("create table regime_daily (date text, regime text)")

    # Two quarters; most recent one should be used and highlighted.
    rows = [
        # 2024Q3
        ("2024-07-01", "risk_off_high_vol"),
        ("2024-07-02", "risk_off_high_vol"),
        ("2024-07-03", "risk_on_low_vol"),
        # 2024Q4 (dominant = risk_on_low_vol)
        ("2024-10-01", "risk_on_low_vol"),
        ("2024-10-02", "risk_on_low_vol"),
        ("2024-10-03", "risk_on_low_vol"),
        ("2024-10-04", "risk_off_mid_vol"),
        ("2024-10-07", "risk_off_mid_vol"),
    ]
    conn.executemany("insert into regime_daily (date, regime) values (?, ?)", rows)

    df = fetch_regime_daily(conn)
    profile, dominant = quarterly_regime_profile(df)

    assert dominant.quarter == "2024Q4"
    assert dominant.dominant_regime == "risk_on_low_vol"
    assert dominant.dominant_days == 3
    assert dominant.total_days == 5

    # Profile contains counts per regime with a dominant highlight.
    expected = (
        pd.DataFrame(
            {
                "quarter": ["2024Q4", "2024Q4"],
                "regime": pd.Series(
                    ["risk_on_low_vol", "risk_off_mid_vol"], dtype="string"
                ),
                "n_days": [3, 2],
                "is_dominant": pd.Series([True, False], dtype="boolean"),
            }
        )
        .sort_values(["n_days", "regime"], ascending=[False, True])
        .reset_index(drop=True)
    )
    got = profile.sort_values(
        ["n_days", "regime"], ascending=[False, True]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, expected)


def test_fetch_market_regime_daily_and_summarize_market_regime_last_3_months_and_trend():
    conn = sqlite3.connect(":memory:")
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

    # 100 calendar days ending 2024-04-10:
    # - In last 90 days: 60 days of "A", 30 days of "B" => dominant "A"
    # - Latest 10 days are "B" => trending regime should be "B"
    dates = pd.date_range("2024-01-02", periods=100, freq="D")
    labels: list[str] = []
    for i in range(len(dates)):
        # last 10 days = B
        if i >= len(dates) - 10:
            labels.append("B")
        else:
            labels.append("A")

    rows = []
    for d, lab in zip(dates, labels):
        rows.append(
            (
                d.date().isoformat(),
                lab,
                1.0 if lab == "A" else 2.0,
                1.0,
                15.0,
                0.0,
                1.0,
                1.0,
                100.0,
                0.0,
                "normal",
                0,
                0,
                None,
                d.date().isoformat(),
            )
        )
    conn.executemany(
        """
        insert into market_regime_daily (
          analysis_date, regime_label, regime_score, position_multiplier,
          vix_level, vix_zscore, vix9d_over_vix, vix_vix3m_ratio,
          move_level, move_zscore, term_structure_state, event_risk_flag,
          joint_stress_flag, notes, created_at
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    df = fetch_market_regime_daily(conn)
    counts, summary = summarize_market_regime(
        df, lookback_days=90, trend_lookback_days=20
    )

    assert summary.dominant_regime == "A"
    assert summary.latest_regime == "B"
    assert summary.latest_streak_days == 10
    assert (
        summary.total_days == 91
    )  # inclusive window end, 90-day lookback yields 91 points in daily calendar series
    assert set(counts["regime"].tolist()) == {"A", "B"}


def test_market_regime_counts_and_trend_defaults_to_10_day_trend_and_fetches_last_90_days():
    conn = sqlite3.connect(":memory:")
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

    # 100 calendar days ending 2024-04-10, with last 10 days = "B"
    dates = pd.date_range("2024-01-02", periods=100, freq="D")
    rows = []
    for i, d in enumerate(dates):
        lab = "B" if i >= len(dates) - 10 else "A"
        rows.append(
            (
                d.date().isoformat(),
                lab,
                1.0 if lab == "A" else 2.0,
                1.0,
                15.0,
                0.0,
                1.0,
                1.0,
                100.0,
                0.0,
                "normal",
                0,
                0,
                None,
                d.date().isoformat(),
            )
        )
    conn.executemany(
        """
        insert into market_regime_daily (
          analysis_date, regime_label, regime_score, position_multiplier,
          vix_level, vix_zscore, vix9d_over_vix, vix_vix3m_ratio,
          move_level, move_zscore, term_structure_state, event_risk_flag,
          joint_stress_flag, notes, created_at
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    df90 = fetch_market_regime_last_n_days(conn, lookback_days=90)
    assert not df90.empty
    # Inclusive endpoints => 91 points for a 90-day lookback window on daily calendar series.
    assert len(df90) == 91

    counts, summary = market_regime_counts_and_trend(conn)
    assert summary.dominant_regime == "A"
    assert summary.latest_regime == "B"
    assert summary.latest_streak_days == 10
    assert summary.total_days == 91
