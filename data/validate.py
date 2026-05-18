# data/validate.py
# post-fetch data quality checks — rejects data that would silently corrupt downstream

import logging

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def validate_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    # 1. columns present
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: missing required columns: {missing}")

    # 2. no duplicate timestamps
    if df["timestamp"].duplicated().any():
        raise ValueError(f"{symbol}: duplicate timestamps found")

    # 3. sorted ascending
    if not df["timestamp"].is_monotonic_increasing:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # 4. high >= low (allow sub-paisa float noise)
    bad_candles = (df["high"] - df["low"]) < -0.01
    if bad_candles.any():
        n_bad = bad_candles.sum()
        raise ValueError(f"{symbol}: {n_bad} rows with high < low")

    # 5. volume non-negative
    if (df["volume"] < 0).any():
        raise ValueError(f"{symbol}: negative volume found")

    # 6. close > 0 for non-null rows
    close = df["close"].dropna()
    if (close <= 0).any():
        raise ValueError(f"{symbol}: close price <= 0 found")

    # 7. at least 1 complete row
    complete = df[REQUIRED_COLUMNS[1:]].notna().all(axis=1)  # exclude timestamp
    if not complete.any():
        raise ValueError(f"{symbol}: no complete ohlcv rows — all NaN")

    logger.info("validated %s: %d rows passed", symbol, len(df))
    return df
