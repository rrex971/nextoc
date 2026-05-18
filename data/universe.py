import json
import logging
import os
import time
from typing import Any

import pandas as pd
from nsepython import nse_eq
from nselib import indices

from config import UNIVERSE_CACHE_PATH

logger = logging.getLogger(__name__)

_RATE_LIMIT_SEC = 0.2


def _rate_limit() -> None:
    time.sleep(_RATE_LIMIT_SEC)


def _market_cap_tier(market_cap: float) -> str:
    # rough sebi-style absolute thresholds for the nifty 500 range
    # values in inr
    if market_cap >= 200_000_000_000:
        return "large_cap"
    elif market_cap >= 50_000_000_000:
        return "mid_cap"
    else:
        return "small_cap"


def _enrich_symbol(symbol: str, company_name: str, sector: str) -> dict | None:
    try:
        data = nse_eq(symbol)
        info = data.get("info", {})
        meta = data.get("metadata", {})
        price_info = data.get("priceInfo", {})
        security_info = data.get("securityInfo", {})

        nse_company_name = info.get("companyName", "") or ""
        industry = meta.get("industry", "") or meta.get("industry", "")
        isin = info.get("isin", "") or ""
        is_fnosec = info.get("isFNOSec", False) or False

        last_price = price_info.get("lastPrice", 0) or 0
        issued_size = security_info.get("issuedSize", 0) or 0
        market_cap = last_price * issued_size

        return {
            "symbol": symbol,
            "company_name": nse_company_name or company_name,
            "sector": sector,
            "industry": industry,
            "market_cap_tier": _market_cap_tier(market_cap),
            "isin": isin,
            "is_fnosec": bool(is_fnosec),
        }
    except Exception:
        logger.warning("nse_eq enrichment failed for %s, skipping", symbol)
        return None


def build_universe() -> list[dict]:
    # step 1: pull the nifty 500 constituent list from nselib
    df = indices.constituent_stock_list(
        index_category="BroadMarketIndices", index_name="Nifty 500"
    )

    universe: list[dict] = []

    for _, row in df.iterrows():
        symbol = str(row.get("Symbol", "")).strip()
        if not symbol:
            continue

        entry = _enrich_symbol(
            symbol=symbol,
            company_name=str(row.get("Company Name", "")),
            sector=str(row.get("Industry", "")),
        )

        if entry is not None:
            universe.append(entry)

        _rate_limit()

    logger.info(
        "built universe: %d symbols enriched from %d nselib rows",
        len(universe),
        len(df),
    )

    # cache write — best effort, don't fail on i/o error
    try:
        _write_cache(universe)
    except OSError:
        logger.error("failed to write universe cache to %s", UNIVERSE_CACHE_PATH)

    return universe


def _write_cache(universe: list[dict]) -> None:
    os.makedirs(os.path.dirname(UNIVERSE_CACHE_PATH) or ".", exist_ok=True)
    with open(UNIVERSE_CACHE_PATH, "w") as f:
        json.dump(universe, f, indent=2)


def _cache_is_stale() -> bool:
    try:
        age = time.time() - os.path.getmtime(UNIVERSE_CACHE_PATH)
        return age > 7 * 24 * 3600
    except FileNotFoundError:
        return True


def load_universe(refresh: bool = False) -> list[dict]:
    if refresh or _cache_is_stale():
        return build_universe()

    try:
        with open(UNIVERSE_CACHE_PATH) as f:
            universe: list[dict] = json.load(f)
        logger.info("loaded %d symbols from universe cache", len(universe))
        return universe
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("universe cache missing or corrupt, rebuilding")
        return build_universe()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    u = load_universe(refresh=True)
    print(f"universe count: {len(u)}")
    for entry in u[:3]:
        print(entry)
