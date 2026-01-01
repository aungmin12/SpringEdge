import sqlite3

from springedge.score_performance import fetch_actionable_score_names, fetch_score_name_groups


def test_fetch_score_name_groups_missing_table_is_empty_not_error():
    conn = sqlite3.connect(":memory:")
    df = fetch_score_name_groups(conn, table="score_performance_evaluation")
    assert df.empty
    assert df.columns.tolist() == ["horizon_days", "regime_label", "n_scores", "score_names"]


def test_fetch_actionable_score_names_filters_all_criteria_and_all_regimes():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "create table score_performance_evaluation (score_name text, horizon_days integer, regime_label text, spearman_ic real, ic_ir real, q5_minus_q1 real)"
    )
    conn.executemany(
        "insert into score_performance_evaluation (score_name, horizon_days, regime_label, spearman_ic, ic_ir, q5_minus_q1) values (?, ?, ?, ?, ?, ?)",
        [
            ("alpha", 365, "risk_on", 0.12, 1.6, 6.0),
            ("alpha", 365, "risk_off", 0.11, 1.7, 5.5),
            ("beta", 365, "risk_on", 0.05, 3.0, 20.0),  # fails A
            ("gamma", 365, "risk_on", 0.20, 1.0, 20.0),  # fails B
            ("delta", 365, "risk_on", 0.20, 2.0, 1.0),  # fails C
            ("epsilon", 21, "risk_on", 0.50, 10.0, 50.0),  # wrong horizon
        ],
    )
    out = fetch_actionable_score_names(conn, table="score_performance_evaluation", horizon_days=365)
    assert out == ["alpha"]

