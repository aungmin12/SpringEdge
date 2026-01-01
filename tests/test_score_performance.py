import sqlite3

from springedge.score_performance import fetch_score_name_groups


def test_fetch_score_name_groups_missing_table_is_empty_not_error():
    conn = sqlite3.connect(":memory:")
    df = fetch_score_name_groups(conn, table="score_performance_evaluation")
    assert df.empty
    assert df.columns.tolist() == ["horizon_days", "regime_label", "n_scores", "score_names"]

