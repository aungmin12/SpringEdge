def test_intelligence_sp500_tickers_importable():
    # The schedule/orchestrator import path must remain stable.
    import intelligence.sp500_tickers as mod

    assert mod.DEFAULT_TABLE == "sp500_tickers"
    assert callable(mod.fetch_sp500_tickers)
