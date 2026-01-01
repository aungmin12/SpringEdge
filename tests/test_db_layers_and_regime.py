import sqlite3

import pandas as pd

from springedge.layers import fetch_sp500_baseline
from springedge.regime import fetch_regime_daily, quarterly_regime_profile


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
                "regime": pd.Series(["risk_on_low_vol", "risk_off_mid_vol"], dtype="string"),
                "n_days": [3, 2],
                "is_dominant": pd.Series([True, False], dtype="boolean"),
            }
        )
        .sort_values(["n_days", "regime"], ascending=[False, True])
        .reset_index(drop=True)
    )
    got = profile.sort_values(["n_days", "regime"], ascending=[False, True]).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, expected)

