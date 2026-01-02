import sqlite3

from springedge.edge import (
    SignalLayerPolicyRow,
    ensure_signal_layer_policy_table,
    score_sp500_from_signal_layer_policy,
    upsert_signal_layer_policy,
)


def test_ensure_signal_layer_policy_table_sqlite_falls_back_when_schema_missing() -> None:
    conn = sqlite3.connect(":memory:")
    tbl = ensure_signal_layer_policy_table(conn, table="intelligence.signal_layer_policy")
    # sqlite without ATTACH should fall back to an unqualified table name.
    assert tbl == "signal_layer_policy"
    # Table should be queryable.
    conn.execute("SELECT * FROM signal_layer_policy WHERE 1=0").fetchall()


def test_score_sp500_from_policy_produces_layer_scores_and_trade_state() -> None:
    conn = sqlite3.connect(":memory:")

    # Baseline universe (matches springedge.layers.fetch_sp500_baseline defaults).
    conn.execute("create table sp500 (date text, symbol text)")
    conn.executemany(
        "insert into sp500 (date, symbol) values (?, ?)",
        [
            ("2024-01-01", "AAA"),
            ("2024-01-01", "BBB"),
        ],
    )

    # Value source (unqualified table; policy scorer retries without schema on sqlite).
    conn.execute(
        """
        create table nonuple_analysis (
          symbol text,
          analysis_date text,
          nonuple_growth real
        )
        """
    )
    conn.executemany(
        "insert into nonuple_analysis (symbol, analysis_date, nonuple_growth) values (?, ?, ?)",
        [
            ("AAA", "2024-01-01", 1.0),
            ("BBB", "2024-01-01", 3.0),
        ],
    )

    # Minimal policy: one quality signal at 365D.
    upsert_signal_layer_policy(
        conn,
        [
            SignalLayerPolicyRow(
                signal_name="nonuple.growth",
                horizon_days=365,
                layer_name="quality_365",
                weight=1.0,
                min_regime="ANY",
                enabled=True,
            )
        ],
        table="intelligence.signal_layer_policy",
    )

    out = score_sp500_from_signal_layer_policy(
        conn,
        baseline_table="sp500",
        baseline_symbol_col="symbol",
        baseline_as_of_col="date",
        baseline_as_of="2024-01-01",
        as_of="2024-01-01",
        policy_table="intelligence.signal_layer_policy",
        quality_threshold=0.0,
        confirm_threshold=0.0,
    )

    assert set(out.columns) >= {
        "symbol",
        "date",
        "layer_z_quality_365",
        "layer_z_confirm_30",
        "layer_z_micro_7",
        "trade_state",
        "edge_score",
    }
    # BBB should be TRADE (positive quality z-score), AAA should be DO_NOT_TRADE.
    states = dict(zip(out["symbol"].astype(str).tolist(), out["trade_state"].astype(str).tolist()))
    assert states["BBB"] == "TRADE"
    assert states["AAA"] == "DO_NOT_TRADE"

