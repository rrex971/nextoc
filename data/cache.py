import datetime as dt
import logging
import sqlite3

import pandas as pd

from config import OHLCV_CACHE_PATH

logger = logging.getLogger(__name__)

_CACHE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _ensure_table(cache_path: str = OHLCV_CACHE_PATH) -> None:
    conn = sqlite3.connect(cache_path)
    try:
        conn.execute(
            "create table if not exists ohlcv_cache ("
            "  symbol    text,"
            "  timestamp text,"
            "  open      real,"
            "  high      real,"
            "  low       real,"
            "  close     real,"
            "  volume    real,"
            "  primary key (symbol, timestamp)"
            ")"
        )
        conn.commit()
    finally:
        conn.close()


def last_cached_date(symbol: str, cache_path: str = OHLCV_CACHE_PATH) -> str | None:
    _ensure_table(cache_path)
    conn = sqlite3.connect(cache_path)
    try:
        row = conn.execute(
            "select max(timestamp) from ohlcv_cache where symbol = ?", (symbol,)
        ).fetchone()
        return row[0] if row and row[0] else None
    finally:
        conn.close()


def cache_is_stale(symbol: str, max_age_days: int = 7, cache_path: str = OHLCV_CACHE_PATH) -> bool:
    last = last_cached_date(symbol, cache_path)
    if last is None:
        return True
    last_date = dt.date.fromisoformat(last)
    return (dt.date.today() - last_date).days > max_age_days


def read_cache(symbol: str, from_date: str, to_date: str, cache_path: str = OHLCV_CACHE_PATH) -> pd.DataFrame | None:
    _ensure_table(cache_path)
    conn = sqlite3.connect(cache_path)
    try:
        rows = conn.execute(
            "select timestamp, open, high, low, close, volume "
            "from ohlcv_cache "
            "where symbol = ? and timestamp >= ? and timestamp <= ? "
            "order by timestamp",
            (symbol, from_date, to_date),
        ).fetchall()
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=_CACHE_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    finally:
        conn.close()


def write_cache(symbol: str, df: pd.DataFrame, cache_path: str = OHLCV_CACHE_PATH) -> None:
    if df.empty:
        return
    _ensure_table(cache_path)
    conn = sqlite3.connect(cache_path)
    try:
        # ensure timestamp column is string for sqlite
        df = df.copy()
        if not isinstance(df["timestamp"].iloc[0], str):
            df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d")
        else:
            # already strings, ensure consistent format
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")

        # confirm required columns are present
        missing = [c for c in _CACHE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"cannot cache {symbol}: missing columns {missing}")

        rows = [tuple([symbol] + [row[c] for c in _CACHE_COLUMNS]) for _, row in df.iterrows()]
        conn.executemany(
            "insert or ignore into ohlcv_cache values (?,?,?,?,?,?,?)", rows
        )
        conn.commit()
        logger.info("cached %d rows for %s", len(rows), symbol)
    finally:
        conn.close()
