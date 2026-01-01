from __future__ import annotations

import numpy as np
import pandas as pd

from springedge import compute_edge_features


def make_toy_ohlcv(
    symbol: str, start: str = "2020-01-01", n: int = 400
) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=n)
    rng = np.random.default_rng(7)
    rets = rng.normal(0, 0.01, size=n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, size=n))
    vol = rng.integers(1_000_000, 5_000_000, size=n)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        }
    )


def main() -> None:
    ohlcv = pd.concat(
        [make_toy_ohlcv("AAPL"), make_toy_ohlcv("MSFT")], ignore_index=True
    )

    # Optional proxies/structural can be joined by symbol/date.
    proxies = ohlcv[["symbol", "date"]].copy()
    proxies["revisions"] = np.nan
    proxies["breadth"] = np.nan
    proxies["price_action_proxy"] = np.nan

    feats = compute_edge_features(ohlcv, proxies=proxies)
    print(feats.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
