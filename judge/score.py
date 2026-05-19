import json
import logging
import sqlite3

from config import LAMBDA

logger = logging.getLogger(__name__)


def parse_alpha(response_json: str) -> tuple[float, str]:
    try:
        parsed = json.loads(response_json)
    except json.JSONDecodeError:
        logger.warning("failed to parse llm response as json: %.80s", response_json)
        return (0.0, "parse error")

    alpha = parsed.get("alpha", 0.0)
    reason = parsed.get("reason", "")

    if not isinstance(alpha, (int, float)):
        logger.warning("alpha is not numeric: %s", alpha)
        return (0.0, reason)

    if not -1.0 <= alpha <= 1.0:
        logger.warning("alpha %.4f out of [-1.0, 1.0], clamping to 0.0", alpha)
        return (0.0, reason)

    return (float(alpha), reason)


def normalize_gamma_scores(candidates: list[dict]) -> list[dict]:
    scores = [c["gamma_score"] for c in candidates]
    g_min = min(scores)
    g_max = max(scores)

    if g_max == g_min:
        for c in candidates:
            c["gamma_normalized"] = 0.5
        return candidates

    for c in candidates:
        c["gamma_normalized"] = (c["gamma_score"] - g_min) / (g_max - g_min)

    return candidates


def final_score(gamma_normalized: float, alpha: float,
                lambda_: float = LAMBDA) -> float:
    return gamma_normalized + lambda_ * alpha


def ensure_llm_triples_table(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            create table if not exists llm_triples (
                id          integer primary key autoincrement,
                date        text not null,
                symbol      text not null,
                news_json   text,
                alpha       real,
                reason      text,
                reward      integer
            )
        """)
        conn.commit()
    finally:
        conn.close()


def write_llm_triple(db_path: str, date: str, symbol: str,
                     news_json: str, alpha: float, reason: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "insert into llm_triples (date, symbol, news_json, alpha, reason) "
            "values (?, ?, ?, ?, ?)",
            (date, symbol, news_json, alpha, reason),
        )
        conn.commit()
    finally:
        conn.close()
