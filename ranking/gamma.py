import logging
import numpy as np
import pandas as pd

from config import (
    LIQUIDITY_FLOOR, HURDLE, TOP_K,
    KELLY_MULTIPLIER, MAX_POSITION_PCT, WIN_PROBABILITY_DEFAULT,
)

logger = logging.getLogger(__name__)


def gamma_score(predicted_return: float, volume: float, price: float,
                liquidity_floor: float = LIQUIDITY_FLOOR) -> float:
    turnover = volume * price
    if turnover < liquidity_floor:
        return float('-inf')
    return predicted_return * np.log1p(turnover)


def kelly_fraction(win_probability: float, predicted_return: float,
                   hurdle: float = HURDLE, multiplier: float = KELLY_MULTIPLIER,
                   max_position_pct: float = MAX_POSITION_PCT) -> float:
    b = predicted_return / hurdle
    if b <= 0:
        return 0.0
    p = win_probability
    q = 1.0 - p
    full_kelly = (b * p - q) / b
    return max(0.0, min(full_kelly * multiplier, max_position_pct))


def rank_by_gamma(
    predictions: dict[str, float],
    symbol_frames: dict[str, pd.DataFrame],
    liquidity_floor: float = LIQUIDITY_FLOOR,
    hurdle: float = HURDLE,
    top_k: int = TOP_K,
    win_probability: float = WIN_PROBABILITY_DEFAULT,
    kelly_multiplier: float = KELLY_MULTIPLIER,
    max_position_pct: float = MAX_POSITION_PCT,
) -> list[dict]:
    candidates = []
    for symbol, pred_return in predictions.items():
        if symbol not in symbol_frames:
            logger.warning("symbol %s not found in symbol_frames, skipping", symbol)
            continue

        if pred_return < hurdle:
            continue

        frame = symbol_frames[symbol]
        latest = frame.iloc[-1]
        close = float(latest["close"])
        volume = int(latest["volume"])

        gs = gamma_score(pred_return, volume, close, liquidity_floor)
        if gs == float('-inf'):
            continue

        kf = kelly_fraction(win_probability, pred_return, hurdle, kelly_multiplier, max_position_pct)
        if kf == 0.0:
            continue

        candidates.append({
            "symbol": symbol,
            "predicted_return": pred_return,
            "close": close,
            "volume": volume,
            "turnover": volume * close,
            "gamma_score": gs,
            "kelly_fraction": kf,
        })

    candidates.sort(key=lambda x: x["gamma_score"], reverse=True)
    return candidates[:top_k]
