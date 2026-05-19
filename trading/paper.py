import logging

from config import HURDLE, MAX_OPEN_POSITIONS, STARTING_PORTFOLIO
from trading.ledger import (
    get_open_trades, get_cash_balance, insert_trade, update_trade_exit,
)

logger = logging.getLogger(__name__)


def _compute_reward(predicted_return: float, actual_return: float,
                    hurdle: float = HURDLE) -> int:
    # directional correctness: predicted must clear the cost hurdle
    predicted_up = predicted_return > hurdle
    actually_up = actual_return > 0
    return 1 if predicted_up == actually_up else -1


def close_positions(
    db_path: str,
    date: str,
    symbol_close_prices: dict[str, float],
    hurdle: float = HURDLE,
) -> int:
    open_trades = get_open_trades(db_path)
    if not open_trades:
        return 0

    closed_count = 0
    for trade in open_trades:
        symbol = trade["symbol"]
        if symbol not in symbol_close_prices:
            logger.warning("close price missing for %s, skipping close", symbol)
            continue

        exit_price = symbol_close_prices[symbol]
        entry_price = trade["entry_price"]
        actual_return = (exit_price - entry_price) / entry_price
        reward = _compute_reward(trade["predicted_return"], actual_return, hurdle)

        update_trade_exit(db_path, trade["id"], exit_price, actual_return, reward)
        closed_count += 1

    return closed_count


def open_positions(
    scored_candidates: list[dict],
    db_path: str,
    date: str,
    regime_flag: int,
    max_positions: int = MAX_OPEN_POSITIONS,
    starting_portfolio: float = STARTING_PORTFOLIO,
) -> int:
    if regime_flag:
        logger.info("regime flag set, skipping new trades")
        return 0

    open_trades = get_open_trades(db_path)
    available_slots = max_positions - len(open_trades)

    if available_slots <= 0:
        return 0

    # portfolio value is cash plus value already deployed in open positions
    cash = get_cash_balance(db_path, starting_portfolio)
    open_value = sum(t["position_size_inr"] for t in open_trades)
    portfolio_value = cash + open_value

    opened_count = 0
    for candidate in scored_candidates[:available_slots]:
        position_size_inr = candidate["kelly_fraction"] * portfolio_value
        insert_trade(
            db_path, date, candidate["symbol"], candidate["sector"],
            candidate["close"], position_size_inr,
            candidate["kelly_fraction"], candidate["predicted_return"],
            candidate["gamma_score"], candidate["alpha"],
            candidate["alpha_reason"], candidate["final_score"],
            regime_flag,
        )
        opened_count += 1

    return opened_count
