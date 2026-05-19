import pandas as pd

from config import MIN_LOOKBACK_DAYS, TARGET_RETURN_CLIP, OUTLIER_SIGMA
from features.crosssectional import compute_crosssectional
from features.indicators import compute_indicators


def daily_return(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change()


def assemble_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return_1d_next"] = (
        df["close"].pct_change()
        .shift(-1)
        .clip(lower=-TARGET_RETURN_CLIP, upper=TARGET_RETURN_CLIP)
    )

    rolling_mean = df["return_1d"].rolling(60).mean()
    rolling_std = df["return_1d"].rolling(60).std()
    z_score = (df["return_1d"] - rolling_mean) / rolling_std
    df["is_outlier"] = (z_score.abs() > OUTLIER_SIGMA).astype(int)

    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    return df


def build_features(
    symbol_frames: dict[str, pd.DataFrame],
    sector_map: dict[str, str],
    cap_tier_map: dict[str, str],
    nifty_return: pd.Series,
) -> dict[str, pd.DataFrame]:
    indicator_frames = {}
    for symbol, df in symbol_frames.items():
        if len(df) < MIN_LOOKBACK_DAYS:
            continue
        df = df.set_index("timestamp")
        indicator_frames[symbol] = compute_indicators(df)

    enriched_frames = compute_crosssectional(
        indicator_frames, sector_map, nifty_return
    )

    result = {}
    for symbol, df in enriched_frames.items():
        df = assemble_features(df)
        df["sector"] = sector_map.get(symbol, "")
        df["market_cap_tier"] = cap_tier_map.get(symbol, "")
        result[symbol] = df

    return result
