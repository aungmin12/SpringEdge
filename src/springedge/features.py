from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from . import indicators as ind
from .layers import calendar_days_to_trading_days


@dataclass(frozen=True)
class EdgeFeatureConfig:
    # regime
    vol_window: int = 20
    vol_regime_low_q: float = 0.33
    vol_regime_high_q: float = 0.67
    risk_on_trend_fast: int = 50
    risk_on_trend_slow: int = 200

    # tactical
    mom_lookbacks: tuple[int, ...] = (21, 63, 126)

    # risk
    atr_window: int = 14
    liq_window: int = 20
    dd_window: int | None = None  # None = since inception

    # forward returns
    fwd_horizons: tuple[int, ...] = (21, 63, 126, 252)
    trading_days: int = 252


REQUIRED_OHLCV_COLS = ("open", "high", "low", "close", "volume")


def _validate_ohlcv_frame(df: pd.DataFrame, *, symbol_col: str, date_col: str) -> None:
    missing = [
        c for c in (symbol_col, date_col, *REQUIRED_OHLCV_COLS) if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _coerce_and_sort(
    df: pd.DataFrame, *, symbol_col: str, date_col: str
) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    for c in REQUIRED_OHLCV_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.sort_values([symbol_col, date_col]).reset_index(drop=True)
    return out


def _vol_regime_id(vol: pd.Series, low_q: float, high_q: float) -> pd.Series:
    """
    0 = low vol, 1 = mid vol, 2 = high vol (per-symbol quantiles).
    """
    lo = vol.quantile(low_q)
    hi = vol.quantile(high_q)
    return pd.Series(
        np.where(
            vol.isna(), np.nan, np.where(vol <= lo, 0.0, np.where(vol >= hi, 2.0, 1.0))
        ),
        index=vol.index,
        dtype="float64",
    )


def _risk_on_flag(trend: pd.Series) -> pd.Series:
    """
    1 = risk-on when trend > 0, 0 = risk-off when trend <= 0
    """
    return pd.Series(
        np.where(trend.isna(), np.nan, np.where(trend > 0, 1.0, 0.0)), index=trend.index
    )


def _regime_label(vol_regime: pd.Series, risk_on_off: pd.Series) -> pd.Series:
    vol_map = {0.0: "low_vol", 1.0: "mid_vol", 2.0: "high_vol"}
    risk_map = {0.0: "risk_off", 1.0: "risk_on"}

    vol_lbl = vol_regime.map(vol_map)
    risk_lbl = risk_on_off.map(risk_map)
    return (risk_lbl.astype("string") + "_" + vol_lbl.astype("string")).where(
        vol_regime.notna() & risk_on_off.notna(), other=pd.NA
    )


def _derive_horizons(
    horizon_days: Iterable[int],
    *,
    basis: Literal["trading", "calendar"],
    trading_days_per_year: int,
) -> list[tuple[int, int]]:
    """
    Returns list of (label_days, lookback_trading_days).
    - label_days: what the user asked for (used in column naming as '{label_days}d')
    - lookback_trading_days: what we compute with in the price series
    """

    out: list[tuple[int, int]] = []
    for d in horizon_days:
        if int(d) <= 0:
            raise ValueError("horizon_days must be positive integers")
        label_days = int(d)
        if basis == "trading":
            lb = label_days
        else:
            lb = calendar_days_to_trading_days(
                label_days, trading_days_per_year=trading_days_per_year
            )
        out.append((label_days, lb))
    # preserve input order, but de-dupe exact pairs
    seen: set[tuple[int, int]] = set()
    deduped: list[tuple[int, int]] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def compute_edge_features(
    ohlcv: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
    universe: str | None = None,
    universe_col: str = "universe",
    config: EdgeFeatureConfig | None = None,
    # Optional: override horizons to create an explicit "Time Horizons" layer
    horizon_days: tuple[int, ...] | None = None,
    horizon_basis: Literal["trading", "calendar"] = "trading",
    # Optional proxy inputs (aligned on symbol/date). Any missing columns => produced as NaN.
    proxies: pd.DataFrame | None = None,
    proxy_cols: Iterable[str] = ("revisions", "breadth", "price_action_proxy"),
    # Optional structural inputs (aligned on symbol/date). Any missing => NaN.
    structural: pd.DataFrame | None = None,
    # This implements a "Nonuple" placeholder set; you can feed real fundamental scores here.
    nonuple_cols: Iterable[str] = (
        "growth",
        "competitive",
        "profitability",
        "balance_sheet",
        "quality",
        "value",
        "sentiment",
        "stability",
        "efficiency",
    ),
) -> pd.DataFrame:
    """
    Build an Edge system feature set per symbol/date.

    Inputs:
    - `ohlcv`: columns [symbol_col, date_col, open, high, low, close, volume]
    - `proxies`: optional columns [symbol_col, date_col, ...proxy_cols]
    - `structural`: optional columns [symbol_col, date_col, ...nonuple_cols] (values should be standardized scores)

    Output columns (minimum):
    - Regime: regime_id, risk_on_off, vol_regime
    - Structural: nonuple_* + composites
    - Tactical: momentum/trend + catalyst/proxies
    - Risk: atr14, ann_vol_20d, drawdown, liquidity, leverage flags
    - Forward returns: fwd_ret_21/63/126/252 (or configured horizons)
    """
    cfg = config or EdgeFeatureConfig()
    _validate_ohlcv_frame(ohlcv, symbol_col=symbol_col, date_col=date_col)
    base = _coerce_and_sort(ohlcv, symbol_col=symbol_col, date_col=date_col)
    if universe_col not in base.columns:
        base[universe_col] = universe
    else:
        # respect any pre-existing values, but allow a global override when explicitly provided
        if universe is not None:
            base[universe_col] = universe

    # Join optional frames
    def _left_join_optional(
        left: pd.DataFrame, right: pd.DataFrame | None
    ) -> pd.DataFrame:
        if right is None:
            return left
        r = right.copy()
        r[date_col] = pd.to_datetime(r[date_col], errors="coerce")
        return left.merge(r, on=[symbol_col, date_col], how="left", suffixes=("", "_y"))

    base = _left_join_optional(base, proxies)
    base = _left_join_optional(base, structural)

    # Ensure proxy/nonuple columns exist even if not provided
    for c in proxy_cols:
        if c not in base.columns:
            base[c] = np.nan
    for c in nonuple_cols:
        if c not in base.columns:
            base[c] = np.nan

    horizons: list[tuple[int, int]] | None = None
    if horizon_days is not None:
        horizons = _derive_horizons(
            horizon_days, basis=horizon_basis, trading_days_per_year=cfg.trading_days
        )

    def per_symbol(g: pd.DataFrame, sym: str) -> pd.DataFrame:
        g = g.sort_values(date_col).copy()
        g[symbol_col] = sym

        # --- Risk primitives
        lr = ind.log_returns(g["close"])
        ann_vol_20d = ind.ann_vol_from_log_returns(
            lr, window=cfg.vol_window, trading_days=cfg.trading_days
        )
        atr14 = ind.atr(g["high"], g["low"], g["close"], window=cfg.atr_window)
        drawdown = ind.max_drawdown(g["close"], window=cfg.dd_window)
        liquidity = (
            (g["close"] * g["volume"])
            .rolling(window=cfg.liq_window, min_periods=cfg.liq_window)
            .mean()
        )

        # --- Tactical: momentum + trend
        trend = ind.trend_ma_cross(
            g["close"], fast=cfg.risk_on_trend_fast, slow=cfg.risk_on_trend_slow
        )
        if horizons is None:
            for lb in cfg.mom_lookbacks:
                g[f"mom_ret_{lb}"] = ind.momentum_return(g["close"], lb)
        else:
            for label_days, lb in horizons:
                g[f"mom_ret_{label_days}d"] = ind.momentum_return(g["close"], lb)
        g["trend_ma_50_200"] = trend

        # --- Regime: vol regime + risk-on/off + regime_id
        vol_regime = _vol_regime_id(
            ann_vol_20d, cfg.vol_regime_low_q, cfg.vol_regime_high_q
        )
        risk_on_off = _risk_on_flag(trend)
        regime = _regime_label(vol_regime, risk_on_off)
        # Combine into a compact regime id: 10*vol_regime + risk_on_off
        regime_id = np.where(
            np.isnan(vol_regime) | np.isnan(risk_on_off),
            np.nan,
            (vol_regime * 10.0 + risk_on_off).astype("float64"),
        )

        g["ann_vol_20d"] = ann_vol_20d
        g["atr14"] = atr14
        g["drawdown"] = drawdown
        g["liquidity"] = liquidity
        g["vol_regime"] = vol_regime
        g["risk_on_off"] = risk_on_off
        g["regime"] = regime
        g["regime_id"] = regime_id

        # --- Structural (Nonuple) + composites
        for c in nonuple_cols:
            g[f"nonuple_{c}"] = pd.to_numeric(g[c], errors="coerce")

        # Example composites (simple averages; expects inputs to be standardized).
        #
        # IMPORTANT: if the caller provided precomputed composite columns (e.g. via the
        # `structural=` frame), do not clobber them with NaNs when nonuple inputs are missing.
        computed_struct_core = g[[f"nonuple_{c}" for c in nonuple_cols]].mean(axis=1)
        computed_struct_growth_quality = g[
            ["nonuple_growth", "nonuple_profitability", "nonuple_quality"]
        ].mean(axis=1)
        computed_struct_value_quality = g[["nonuple_value", "nonuple_quality"]].mean(
            axis=1
        )
        if "struct_core" in g.columns:
            existing = pd.to_numeric(g["struct_core"], errors="coerce")
            g["struct_core"] = existing.where(existing.notna(), computed_struct_core)
        else:
            g["struct_core"] = computed_struct_core
        if "struct_growth_quality" in g.columns:
            existing = pd.to_numeric(g["struct_growth_quality"], errors="coerce")
            g["struct_growth_quality"] = existing.where(
                existing.notna(), computed_struct_growth_quality
            )
        else:
            g["struct_growth_quality"] = computed_struct_growth_quality
        if "struct_value_quality" in g.columns:
            existing = pd.to_numeric(g["struct_value_quality"], errors="coerce")
            g["struct_value_quality"] = existing.where(
                existing.notna(), computed_struct_value_quality
            )
        else:
            g["struct_value_quality"] = computed_struct_value_quality

        # --- Risk: leverage flags (optional)
        # If user provides e.g. debt_to_equity, net_debt_to_ebitda, etc., we can flag here.
        # For now, expose a generic placeholder flag column.
        g["leverage_flag"] = np.nan

        # --- Tactical: catalyst / proxies passthrough
        g["revisions"] = pd.to_numeric(g["revisions"], errors="coerce")
        g["breadth"] = pd.to_numeric(g["breadth"], errors="coerce")
        g["price_action_proxy"] = pd.to_numeric(
            g["price_action_proxy"], errors="coerce"
        )

        # --- Forward returns
        if horizons is None:
            for h in cfg.fwd_horizons:
                g[f"fwd_ret_{h}"] = ind.forward_return(g["close"], h)
        else:
            for label_days, lb in horizons:
                g[f"fwd_ret_{label_days}d"] = ind.forward_return(g["close"], lb)

        return g

    # Use an explicit group loop to avoid pandas GroupBy.apply behavioral changes
    # and to keep the output schema stable across pandas versions.
    parts: list[pd.DataFrame] = []
    for sym, g in base.groupby(symbol_col, sort=False):
        parts.append(per_symbol(g, sym))
    out = pd.concat(parts, ignore_index=True)

    # Final column ordering: keep original keys first, then feature families
    base_cols = [symbol_col, universe_col, date_col, *REQUIRED_OHLCV_COLS]
    feature_cols = [
        c for c in out.columns if c not in base_cols and c not in set(nonuple_cols)
    ]
    out = out[base_cols + feature_cols]
    return out
