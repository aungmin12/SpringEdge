"""
Lightweight schema/table registry.

SpringEdge intentionally avoids heavyweight schema management (SQLAlchemy/Alembic).
This module is a small, explicit inventory of upstream tables that other systems
may reference.
"""

from __future__ import annotations

from typing import Final, Iterable

CORE_SCHEMA: Final[str] = "core"
INTELLIGENCE_SCHEMA: Final[str] = "intelligence"

# Core schema (canonical base data).
CORE_TABLES: Final[tuple[str, ...]] = (
    "prices_daily",
    "security",
    "sp500_tickers",
)

# Intelligence schema (derived/intelligence datasets).
INTELLIGENCE_TABLES: Final[tuple[str, ...]] = (
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
    "security_universe",
    "trade_phase_tracking",
    "trade_readiness_scores",
    "trs_guardrails",
)


def qualify_table(schema: str, table: str) -> str:
    """
    Return a schema-qualified table reference: "schema.table".
    """

    s = str(schema or "").strip()
    t = str(table or "").strip()
    if not s:
        raise ValueError("schema must be non-empty")
    if not t:
        raise ValueError("table must be non-empty")
    return f"{s}.{t}"


def iter_all_tables(*, include_schema: bool = False) -> Iterable[str]:
    """
    Iterate known tables across all schemas.

    - include_schema=False: yields bare table names.
    - include_schema=True: yields schema-qualified names (e.g. "core.prices_daily").
    """

    if include_schema:
        yield from (qualify_table(CORE_SCHEMA, t) for t in CORE_TABLES)
        yield from (qualify_table(INTELLIGENCE_SCHEMA, t) for t in INTELLIGENCE_TABLES)
    else:
        yield from CORE_TABLES
        yield from INTELLIGENCE_TABLES


__all__ = [
    "CORE_SCHEMA",
    "INTELLIGENCE_SCHEMA",
    "CORE_TABLES",
    "INTELLIGENCE_TABLES",
    "qualify_table",
    "iter_all_tables",
]

