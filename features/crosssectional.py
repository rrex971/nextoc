import pandas as pd

from config import REGIME_WINDOW, REGIME_Z_THRESHOLD


def compute_regime_flag(nifty_returns: pd.Series, window: int = REGIME_WINDOW) -> pd.Series:
    rolling_mean = nifty_returns.rolling(window).mean()
    rolling_std = nifty_returns.rolling(window).std()
    z_score = (nifty_returns - rolling_mean) / rolling_std
    return (z_score < REGIME_Z_THRESHOLD).astype(int)


def compute_crosssectional(
    frames: dict[str, pd.DataFrame],
    sector_map: dict[str, str],
    nifty_return: pd.Series,
    window: int = REGIME_WINDOW,
) -> dict[str, pd.DataFrame]:
    returns = {}
    for symbol, df in frames.items():
        returns[symbol] = df["close"].pct_change()

    returns_wide = pd.DataFrame(returns)
    nifty_aligned = nifty_return.reindex(returns_wide.index)

    sector_symbols: dict[str, list[str]] = {}
    for symbol, sector in sector_map.items():
        if symbol in frames:
            sector_symbols.setdefault(sector, []).append(symbol)

    sector_means = {}
    for sector, syms in sector_symbols.items():
        valid = [s for s in syms if s in returns_wide.columns]
        if valid:
            sector_means[sector] = returns_wide[valid].mean(axis=1)

    sector_return_df = pd.DataFrame(sector_means)
    regime = compute_regime_flag(nifty_aligned, window=window)

    result = {}
    for symbol, df in frames.items():
        enriched = df.copy()
        sector = sector_map.get(symbol, "")
        ret = returns[symbol]

        enriched["return_1d"] = ret
        enriched["sector_return_1d"] = sector_return_df[sector]
        enriched["stock_vs_sector"] = ret - enriched["sector_return_1d"]
        enriched["nifty50_return_1d"] = nifty_aligned
        enriched["stock_vs_market"] = ret - nifty_aligned
        enriched["regime_flag"] = regime

        result[symbol] = enriched

    return result
