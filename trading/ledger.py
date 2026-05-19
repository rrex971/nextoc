import logging
import sqlite3

logger = logging.getLogger(__name__)


def ensure_trades_table(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            create table if not exists trades (
                id                integer primary key autoincrement,
                date              text not null,
                symbol            text not null,
                sector            text,
                entry_price       real,
                exit_price        real,
                position_size_inr real,
                kelly_fraction    real,
                predicted_return  real,
                gamma_score       real,
                alpha             real,
                alpha_reason      text,
                final_score       real,
                actual_return     real,
                reward            integer,
                regime_flag       integer
            )
        """)
        conn.commit()
    finally:
        conn.close()


def insert_trade(
    db_path: str, date: str, symbol: str, sector: str,
    entry_price: float, position_size_inr: float,
    kelly_fraction: float, predicted_return: float,
    gamma_score: float, alpha: float, alpha_reason: str,
    final_score: float, regime_flag: int,
) -> int:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "insert into trades (date, symbol, sector, entry_price, "
            "position_size_inr, kelly_fraction, predicted_return, "
            "gamma_score, alpha, alpha_reason, final_score, regime_flag) "
            "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (date, symbol, sector, entry_price, position_size_inr,
             kelly_fraction, predicted_return, gamma_score, alpha,
             alpha_reason, final_score, regime_flag),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def update_trade_exit(
    db_path: str, trade_id: int, exit_price: float,
    actual_return: float, reward: int,
) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "update trades set exit_price = ?, actual_return = ?, reward = ? "
            "where id = ?",
            (exit_price, actual_return, reward, trade_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_open_trades(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "select * from trades where exit_price is null"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_closed_trades(db_path: str, limit_days: int = 30) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "select * from trades where exit_price is not null "
            "order by date desc limit ?",
            (limit_days,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_cash_balance(db_path: str, starting_portfolio: float) -> float:
    # realized pnl = sum of position_size_inr * actual_return for closed trades
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "select position_size_inr, actual_return from trades "
            "where exit_price is not null and actual_return is not null"
        ).fetchall()
    finally:
        conn.close()

    realized_pnl = sum(row[0] * row[1] for row in rows)
    return starting_portfolio + realized_pnl
