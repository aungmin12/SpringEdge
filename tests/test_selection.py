import sqlite3

from springedge.selection import pick_qualified_stocks


def test_pick_qualified_stocks_uses_score_name_columns_and_direction() -> None:
    conn = sqlite3.connect(":memory:")

    # Performance metadata table (what's "actionable")
    conn.execute(
        """
        create table score_performance_evaluation (
          score_name text,
          horizon_days integer,
          regime_label text,
          spearman_ic real,
          ic_ir real,
          q5_minus_q1 real
        )
        """
    )
    # rsi: positive IC => higher is better (top quantile should pass)
    # debt_to_equity: negative IC => lower is better (bottom quantile should pass)
    conn.executemany(
        """
        insert into score_performance_evaluation
          (score_name, horizon_days, regime_label, spearman_ic, ic_ir, q5_minus_q1)
        values (?, ?, ?, ?, ?, ?)
        """,
        [
            ("rsi", 365, "risk_on", 0.20, 2.0, 6.0),
            ("debt_to_equity", 365, "risk_on", -0.20, 2.0, 6.0),
        ],
    )

    # Value sources in intelligence schema (simulate via unqualified names for sqlite)
    conn.execute(
        """
        create table nonuple_technical_indicators (
          symbol text,
          analysis_date text,
          rsi real
        )
        """
    )
    conn.execute(
        """
        create table nonuple_financial_metrics (
          symbol text,
          analysis_date text,
          debt_to_equity real
        )
        """
    )
    conn.executemany(
        "insert into nonuple_technical_indicators (symbol, analysis_date, rsi) values (?, ?, ?)",
        [
            ("AAA", "2024-01-01", 10.0),
            ("BBB", "2024-01-01", 50.0),
            ("CCC", "2024-01-01", 90.0),
        ],
    )
    conn.executemany(
        "insert into nonuple_financial_metrics (symbol, analysis_date, debt_to_equity) values (?, ?, ?)",
        [
            ("AAA", "2024-01-01", 3.0),
            ("BBB", "2024-01-01", 1.0),
            ("CCC", "2024-01-01", 0.1),
        ],
    )

    out = pick_qualified_stocks(
        conn,
        horizon_days=365,
        as_of="2024-01-01",
        # For sqlite tests, pass sources without schema qualification.
        sources=[
            # match the default date/symbol columns we used above
            # (the function's defaults are intelligence.* which sqlite won't have)
            # so we explicitly pass these.
            # rsi:
            # debt_to_equity:
            # (two sources)
            # NOTE: we keep the default col names.
            __import__("springedge.selection").selection.ScoreValueSource(
                "nonuple_technical_indicators", "symbol", "analysis_date"
            ),
            __import__("springedge.selection").selection.ScoreValueSource(
                "nonuple_financial_metrics", "symbol", "analysis_date"
            ),
        ],
        pass_quantile=2 / 3,  # with 3 symbols, top/bottom 1 name passes
        min_pass_fraction=1.0,  # require passing both signals
        return_details=True,
    )

    # Expect CCC: highest rsi AND lowest debt_to_equity.
    assert out["symbol"].tolist() == ["CCC"]
    assert out["pass_count"].iloc[0] == 2


def test_fetch_latest_score_values_with_aliases_supports_dotted_score_names() -> None:
    conn = sqlite3.connect(":memory:")

    # Physical table uses identifier-safe underscore column.
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
            ("BBB", "2024-01-01", 2.0),
        ],
    )

    from springedge.selection import ScoreValueSource, fetch_latest_score_values_with_aliases

    out = fetch_latest_score_values_with_aliases(
        conn,
        scores=["nonuple.growth"],
        sources=[ScoreValueSource("nonuple_analysis", "symbol", "analysis_date")],
        as_of="2024-01-01",
    )

    # Output column name should be the requested dotted score name.
    assert out.columns.tolist() == ["symbol", "date", "nonuple.growth"]
    assert out.sort_values("symbol")["nonuple.growth"].tolist() == [1.0, 2.0]

