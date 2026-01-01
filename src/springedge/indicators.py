from __future__ import annotations

import numpy as np
import pandas as pd


def _as_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    True range:
      TR_t = max( high-low, abs(high-prev_close), abs(low-prev_close) )
    """
    h = _as_float(high)
    l = _as_float(low)
    c = _as_float(close)
    prev_c = c.shift(1)
    a = (h - l).abs()
    b = (h - prev_c).abs()
    c_ = (l - prev_c).abs()
    return pd.concat([a, b, c_], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Simple moving average ATR on true range."""
    tr = true_range(high, low, close)
    return tr.rolling(window=window, min_periods=window).mean()


def log_returns(close: pd.Series) -> pd.Series:
    c = _as_float(close)
    return np.log(c).diff()


def ann_vol_from_log_returns(lr: pd.Series, window: int = 20, trading_days: int = 252) -> pd.Series:
    """Annualized volatility using rolling std of log returns."""
    x = _as_float(lr)
    return x.rolling(window=window, min_periods=window).std() * np.sqrt(trading_days)


def max_drawdown(close: pd.Series, window: int | None = None) -> pd.Series:
    """
    Drawdown (negative or 0):
      dd_t = close_t / rolling_max(close) - 1
    If window is None, uses expanding max.
    """
    c = _as_float(close)
    if window is None:
        peak = c.cummax()
    else:
        peak = c.rolling(window=window, min_periods=1).max()
    return c / peak - 1.0


def sma(series: pd.Series, window: int) -> pd.Series:
    s = _as_float(series)
    return s.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    s = _as_float(series)
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def zscore(series: pd.Series, window: int) -> pd.Series:
    s = _as_float(series)
    mean = s.rolling(window=window, min_periods=window).mean()
    std = s.rolling(window=window, min_periods=window).std()
    return (s - mean) / std.replace(0.0, np.nan)


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """
    Simple forward return:
      fwd_ret_h = close_{t+h}/close_t - 1
    """
    c = _as_float(close)
    return c.shift(-horizon) / c - 1.0


def momentum_return(close: pd.Series, lookback: int) -> pd.Series:
    """Simple trailing return over lookback days."""
    c = _as_float(close)
    return c / c.shift(lookback) - 1.0


def trend_ma_cross(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """Trend proxy: fast SMA minus slow SMA (normalized by close)."""
    c = _as_float(close)
    fast_ma = sma(c, fast)
    slow_ma = sma(c, slow)
    return (fast_ma - slow_ma) / c
