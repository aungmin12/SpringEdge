import sqlite3

from springedge.score_performance import fetch_actionable_score_names, fetch_average_returns_by_horizon, fetch_score_name_groups


def test_fetch_score_name_groups_missing_table_is_empty_not_error():
    conn = sqlite3.connect(":memory:")
    df = fetch_score_name_groups(conn, table="score_performance_evaluation")
    assert df.empty
    assert df.columns.tolist() == [
        "horizon_days",
        "regime_label",
        "n_scores",
        "score_names",
    ]


def test_fetch_average_returns_by_horizon_groups_and_rounds():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "create table score_performance_evaluation (horizon_days integer, mean_return real, median_return real)"
    )
    conn.executemany(
        "insert into score_performance_evaluation (horizon_days, mean_return, median_return) values (?, ?, ?)",
        [
            (7, 1.2, 1.4),
            (7, 1.6, 1.1),
            (21, 10.1, 9.9),
            (21, 9.9, 10.2),
        ],
    )
    df = fetch_average_returns_by_horizon(conn, table="score_performance_evaluation")
    assert df.columns.tolist() == ["horizon_days", "avg_mean_return", "avg_median_return"]

    # 7d: mean(mean_return)=1.4 -> round=1; mean(median_return)=1.25 -> round=1
    # 21d: mean(mean_return)=10.0 -> round=10; mean(median_return)=10.05 -> round=10
    assert df.to_dict(orient="records") == [
        {"horizon_days": 7, "avg_mean_return": 1.0, "avg_median_return": 1.0},
        {"horizon_days": 21, "avg_mean_return": 10.0, "avg_median_return": 10.0},
    ]


def test_fetch_actionable_score_names_filters_all_criteria_and_all_regimes():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "create table score_performance_evaluation (score_name text, horizon_days integer, regime_label text, spearman_ic real, ic_ir real, q5_minus_q1 real)"
    )
    conn.executemany(
        "insert into score_performance_evaluation (score_name, horizon_days, regime_label, spearman_ic, ic_ir, q5_minus_q1) values (?, ?, ?, ?, ?, ?)",
        [
            ("alpha", 365, "risk_on", 0.12, 1.6, 0.06),
            ("alpha", 365, "risk_off", 0.11, 1.7, 0.055),
            ("beta", 365, "risk_on", 0.05, 3.0, 0.20),  # fails A
            ("gamma", 365, "risk_on", 0.20, 1.0, 0.20),  # fails B
            ("delta", 365, "risk_on", 0.20, 2.0, 0.01),  # fails C
            ("epsilon", 21, "risk_on", 0.50, 10.0, 0.50),  # wrong horizon
            # Passes thresholds in one regime but fails in another; should be excluded by default (ALL regimes).
            ("zeta", 365, "risk_on", 0.20, 2.0, 0.10),
            ("zeta", 365, "risk_off", 0.01, 2.0, 0.10),  # fails A in this regime
        ],
    )
    out = fetch_actionable_score_names(
        conn, table="score_performance_evaluation", horizon_days=365
    )
    assert out == ["alpha"]


def test_fetch_actionable_score_names_any_regime_matches_distinct_row_filtering():
    """
    SQL like:
      SELECT DISTINCT score_name FROM ... WHERE <thresholds>
    is equivalent to "any regime row passes" when regime rows exist.
    """
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "create table score_performance_evaluation (score_name text, horizon_days integer, regime_label text, spearman_ic real, ic_ir real, q5_minus_q1 real)"
    )
    conn.executemany(
        "insert into score_performance_evaluation (score_name, horizon_days, regime_label, spearman_ic, ic_ir, q5_minus_q1) values (?, ?, ?, ?, ?, ?)",
        [
            ("alpha", 365, "risk_on", 0.12, 1.6, 0.06),
            ("alpha", 365, "risk_off", 0.11, 1.7, 0.055),
            ("zeta", 365, "risk_on", 0.20, 2.0, 0.10),
            ("zeta", 365, "risk_off", 0.01, 2.0, 0.10),  # fails A in this regime
        ],
    )
    out = fetch_actionable_score_names(
        conn,
        table="score_performance_evaluation",
        horizon_days=365,
        require_all_regimes=False,
    )
    assert out == ["alpha", "zeta"]


def test_fetch_actionable_score_names_sql_failure_rolls_back_and_falls_back_to_df():
    """
    Regression: if the SQL pushdown path fails on Postgres, psycopg will mark the
    transaction as aborted until rollback(). The fallback pandas path must still work.
    """

    class _FakeCursor:
        def __init__(self, conn: "_FakePsycopgConn") -> None:
            self._conn = conn
            self.description = None

        def execute(self, sql: str, params=None) -> None:  # noqa: ANN001 - test stub
            # First attempt: SQL pushdown fails and aborts txn.
            if params is not None and not self._conn._attempted_pushdown:
                self._conn._attempted_pushdown = True
                self._conn._aborted = True
                raise RuntimeError("simulated pushdown SQL error")

            # Any query while aborted should raise the typical psycopg behavior.
            if self._conn._aborted:
                raise RuntimeError(
                    "current transaction is aborted, commands ignored until end of transaction block"
                )

            # Fallback SELECT succeeds.
            self.description = [
                ("score_name", None, None, None, None, None, None),
                ("horizon_days", None, None, None, None, None, None),
                ("regime_label", None, None, None, None, None, None),
                ("spearman_ic", None, None, None, None, None, None),
                ("ic_ir", None, None, None, None, None, None),
                ("q5_minus_q1", None, None, None, None, None, None),
            ]

        def fetchall(self):
            return [
                ("alpha", 365, "risk_on", 0.12, 1.6, 0.06),
                ("alpha", 365, "risk_off", 0.11, 1.7, 0.055),
                ("beta", 365, "risk_on", 0.05, 3.0, 0.20),
            ]

        def close(self) -> None:
            return None

    class _FakePsycopgConn:
        __module__ = "psycopg.connection"

        def __init__(self) -> None:
            self._aborted = False
            self._attempted_pushdown = False

        def cursor(self) -> _FakeCursor:
            return _FakeCursor(self)

        def rollback(self) -> None:
            self._aborted = False

    conn = _FakePsycopgConn()
    out = fetch_actionable_score_names(
        conn, table="score_performance_evaluation", horizon_days=365
    )
    assert out == ["alpha"]
