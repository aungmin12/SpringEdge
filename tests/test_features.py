import pandas as pd

from springedge import compute_edge_features


def test_compute_edge_features_basic_schema():
    df = pd.DataFrame(
        {
            "symbol": ["AAA"] * 260,
            "date": pd.bdate_range("2020-01-01", periods=260),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": pd.Series(range(260), dtype="float64") + 100.0,
            "volume": 1_000_000,
        }
    )
    out = compute_edge_features(df)

    # Regime
    for c in ["regime_id", "risk_on_off", "vol_regime"]:
        assert c in out.columns

    # Tactical
    for c in ["trend_ma_50_200", "mom_ret_21", "mom_ret_63", "mom_ret_126"]:
        assert c in out.columns

    # Risk
    for c in ["atr14", "ann_vol_20d", "drawdown", "liquidity", "leverage_flag"]:
        assert c in out.columns

    # Forward returns
    for c in ["fwd_ret_21", "fwd_ret_63", "fwd_ret_126", "fwd_ret_252"]:
        assert c in out.columns


def test_compute_edge_features_multi_symbol_grouping():
    a = pd.DataFrame(
        {
            "symbol": ["A"] * 60,
            "date": pd.bdate_range("2021-01-01", periods=60),
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "close": 10.0,
            "volume": 100,
        }
    )
    b = a.copy()
    b["symbol"] = "B"
    b["close"] = 20.0
    df = pd.concat([a, b], ignore_index=True)

    out = compute_edge_features(df)
    assert set(out["symbol"].unique()) == {"A", "B"}
    assert len(out) == len(df)
