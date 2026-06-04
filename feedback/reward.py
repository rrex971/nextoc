import logging

from config import HURDLE, WIN_PROBABILITY_DEFAULT, WIN_PROBABILITY_MIN_TRADES
from trading.ledger import get_closed_trades

logger = logging.getLogger(__name__)


def compute_reward(predicted_return: float, actual_return: float,
                   hurdle: float = HURDLE) -> int:
    predicted_up = predicted_return > hurdle
    actually_up = actual_return > 0
    return 1 if predicted_up == actually_up else -1


def compute_rolling_accuracy(db_path: str, lookback: int = 30) -> float:
    closed = get_closed_trades(db_path, lookback)
    if len(closed) < WIN_PROBABILITY_MIN_TRADES:
        return WIN_PROBABILITY_DEFAULT

    wins = sum(1 for t in closed if t.get("reward") == 1)
    return wins / len(closed)
