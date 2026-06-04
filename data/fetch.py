import logging
import datetime as dt
import time
from typing import Any

import pandas as pd
from growwapi import GrowwAPI
from growwapi.groww.exceptions import GrowwAPIException

from config import GROWW_API_KEY, GROWW_API_SECRET
from data.cache import cache_is_stale, last_cached_date, read_cache, write_cache
from data.validate import validate_ohlcv

logger = logging.getLogger(__name__)

_token: str | None = None
_client: GrowwAPI | None = None
_token_date: dt.date | None = None

_RATE_LIMIT_SEC = 0.05
_MAX_RETRY_ATTEMPTS = 3
_BACKOFF_BASE_DELAY = 1.0


def _get_client() -> GrowwAPI:
    global _token, _client, _token_date
    today = dt.date.today()
    if _client is not None and _token_date == today:
        return _client
    _token = GrowwAPI.get_access_token(api_key=GROWW_API_KEY, secret=GROWW_API_SECRET)
    _client = GrowwAPI(_token)
    _token_date = today
    logger.info("groww access token regenerated")
    return _client


def _groww_symbol(symbol: str) -> str:
    return f"NSE-{symbol.upper()}"


def _datetime_range(from_date: str, to_date: str) -> tuple[str, str]:
    return f"{from_date} 00:00:00", f"{to_date} 23:59:59"


def _parse_candle(candle: list | dict[str, Any]) -> dict[str, Any]:
    if isinstance(candle, list):
        return {
            "timestamp": candle[0],
            "open": candle[1],
            "high": candle[2],
            "low": candle[3],
            "close": candle[4],
            "volume": candle[5],
        }
    return {
        "timestamp": candle["timestamp"],
        "open": candle["open"],
        "high": candle["high"],
        "low": candle["low"],
        "close": candle["close"],
        "volume": candle["volume"],
    }


def _candles_to_dataframe(candles: list) -> pd.DataFrame:
    normalized = [_parse_candle(c) for c in candles]
    df = pd.DataFrame(normalized)
    if df["timestamp"].dtype in ("int64", "float64"):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def _fetch_from_groww(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    client = _get_client()
    start_time, end_time = _datetime_range(from_date, to_date)
    g_symbol = _groww_symbol(symbol)

    response = client.get_historical_candles(
        exchange=GrowwAPI.EXCHANGE_NSE,
        segment=GrowwAPI.SEGMENT_CASH,
        groww_symbol=g_symbol,
        start_time=start_time,
        end_time=end_time,
        candle_interval=GrowwAPI.CANDLE_INTERVAL_DAY,
        timeout=30,
    )

    candles = response.get("candles", [])
    if not candles:
        raise ValueError(f"no candle data returned for {symbol} ({from_date} to {to_date})")

    df = _candles_to_dataframe(candles)
    logger.info("fetched %d trading days for %s", len(df), symbol)
    return df


def _retry_with_backoff(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    global _client, _token_date
    for attempt in range(_MAX_RETRY_ATTEMPTS):
        try:
            return _fetch_from_groww(symbol, from_date, to_date)
        except GrowwAPIException as e:
            if hasattr(e, "code") and e.code in ("GA005",):
                _client = None
                _token_date = None
            delay = _BACKOFF_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "groww api attempt %d/%d for %s failed: %s, retrying in %.1fs",
                attempt + 1, _MAX_RETRY_ATTEMPTS, symbol, e, delay,
            )
            if attempt < _MAX_RETRY_ATTEMPTS - 1:
                time.sleep(delay)
                continue
            raise
    raise RuntimeError(f"unreachable: retry loop exhausted for {symbol}")


def fetch_ohlcv(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    # check if a full refetch is needed
    if cache_is_stale(symbol):
        logger.info("cache stale for %s, fetching full range", symbol)
        time.sleep(_RATE_LIMIT_SEC)
        df = _retry_with_backoff(symbol, from_date, to_date)
        write_cache(symbol, df)
    else:
        last = last_cached_date(symbol)
        if last is not None and last >= to_date:
            cached = read_cache(symbol, from_date, to_date)
            if cached is not None and not cached.empty:
                logger.info("cache hit for %s (%d rows)", symbol, len(cached))
                df = cached
            else:
                time.sleep(_RATE_LIMIT_SEC)
                df = _retry_with_backoff(symbol, from_date, to_date)
                write_cache(symbol, df)
        else:
            # fetch only missing days
            if last is not None:
                fetch_from = (dt.date.fromisoformat(last) + dt.timedelta(days=1)).isoformat()
            else:
                fetch_from = from_date
            logger.info("partial fetch for %s: %s to %s", symbol, fetch_from, to_date)
            time.sleep(_RATE_LIMIT_SEC)
            new_df = _retry_with_backoff(symbol, fetch_from, to_date)
            write_cache(symbol, new_df)

            cached_df = read_cache(symbol, from_date, to_date)
            if cached_df is not None and not cached_df.empty:
                df = pd.concat([cached_df, new_df], ignore_index=True)
                df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
            else:
                df = new_df

    # validate before returning
    try:
        df = validate_ohlcv(df, symbol)
    except Exception:
        logger.warning("validation failed for %s, returning raw data", symbol)
    return df
