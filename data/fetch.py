# data/fetch.py
# groww api wrapper — fetches daily ohlcv data for nse equities
# wraps growwapi.GrowwAPI with token lifecycle management and response normalization

import logging
import datetime as dt
from typing import Any

import pandas as pd
from growwapi import GrowwAPI
from growwapi.groww.exceptions import GrowwAPIException

from config import GROWW_API_KEY, GROWW_API_SECRET

logger = logging.getLogger(__name__)

_token: str | None = None
_client: GrowwAPI | None = None
_token_date: dt.date | None = None


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
    # candles come as either array[array] or array[object] depending on endpoint version
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
    # api returns iso strings, not unix epochs; handle both
    if df["timestamp"].dtype in ("int64", "float64"):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def fetch_ohlcv(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    client = _get_client()
    start_time, end_time = _datetime_range(from_date, to_date)
    g_symbol = _groww_symbol(symbol)

    try:
        response = client.get_historical_candles(
            exchange=GrowwAPI.EXCHANGE_NSE,
            segment=GrowwAPI.SEGMENT_CASH,
            groww_symbol=g_symbol,
            start_time=start_time,
            end_time=end_time,
            candle_interval=GrowwAPI.CANDLE_INTERVAL_DAY,
            timeout=30,
        )
    except GrowwAPIException as e:
        # if token expired (GA005), regenerate and retry once
        if hasattr(e, "code") and e.code in ("GA005",):
            global _client, _token_date
            _client = None
            _token_date = None
            client = _get_client()
            response = client.get_historical_candles(
                exchange=GrowwAPI.EXCHANGE_NSE,
                segment=GrowwAPI.SEGMENT_CASH,
                groww_symbol=g_symbol,
                start_time=start_time,
                end_time=end_time,
                candle_interval=GrowwAPI.CANDLE_INTERVAL_DAY,
                timeout=30,
            )
        else:
            logger.error("groww api error for %s: %s", symbol, e)
            raise

    candles = response.get("candles", [])
    if not candles:
        raise ValueError(f"no candle data returned for {symbol} ({from_date} to {to_date})")

    df = _candles_to_dataframe(candles)
    logger.info("fetched %d trading days for %s", len(df), symbol)
    return df
