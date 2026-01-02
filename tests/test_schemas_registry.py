from springedge.schemas import (
    CORE_SCHEMA,
    CORE_TABLES,
    INTELLIGENCE_SCHEMA,
    INTELLIGENCE_TABLES,
    iter_all_tables,
    qualify_table,
)


def test_schema_registry_contains_expected_tables() -> None:
    assert CORE_SCHEMA == "core"
    assert INTELLIGENCE_SCHEMA == "intelligence"

    assert set(CORE_TABLES) == {"prices_daily", "security", "sp500_tickers"}

    expected_intelligence = {
        "company_catalysts",
        "dark_pool_accumulation_scores",
        "earnings_edge_scores",
        "earnings_news_intelligence",
        "earnings_surprise_tape_analysis",
        "enhanced_insider_intelligence",
        "estimate_revisions",
        "macro_series",
        "market_breadth_core_daily",
        "market_regime_assessment",
        "market_regime_daily",
        "mid_quarter_performance_scores",
        "nonuple_analysis",
        "nonuple_financial_metrics",
        "nonuple_technical_indicators",
        "nonuple_workflow_state",
        "options_flow_enhanced_scores_v2",
        "options_flow_enhanced_v2",
        "options_flow_intelligence",
        "options_flow_summary",
        "options_positioning_detail",
        "ownership_intelligence",
        "ownership_positioning",
        "pre_earnings_intelligence_score",
        "ray_dalio_empire_analysis",
        "risk_calendar_events",
        "score_performance_evaluation",
        "signal_layer_policy",
        "security_universe",
        "trade_phase_tracking",
        "trade_readiness_scores",
        "trs_guardrails",
    }
    assert set(INTELLIGENCE_TABLES) == expected_intelligence

    # Smoke-check helpers.
    assert qualify_table(CORE_SCHEMA, "prices_daily") == "core.prices_daily"
    all_bare = set(iter_all_tables(include_schema=False))
    all_qualified = set(iter_all_tables(include_schema=True))
    assert all_bare == set(CORE_TABLES) | set(INTELLIGENCE_TABLES)
    assert "core.prices_daily" in all_qualified
    assert "intelligence.company_catalysts" in all_qualified
