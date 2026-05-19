import json
import logging

from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MAX_TOKENS, LLM_RETRY_ATTEMPTS
from judge.score import parse_alpha

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """you are a financial analyst reviewing nse-listed indian equities for next-day intraday trading.

you will receive:
- stock symbol and sector
- predicted next-day return from a quantitative model
- key technical indicator snapshot
- recent news articles retrieved via search (title, snippet, published time)
- current regime_flag (0 = normal market, 1 = stress/crash conditions)

(the model's attention weights are not available in this version.)

your task: produce a single alpha score in [-1.0, 1.0].

rules:
- ground your score only in the provided news. do not fabricate events.
- if no relevant news was found, return 0.0 exactly. do not guess.
- news older than 48 hours for a fast-moving event should be heavily discounted.
- if regime_flag = 1, bias toward 0.0 or negative regardless of news — the model's predictions are unreliable in stress regimes.
- provide one short sentence explaining your score.

respond with json only. no prose, no markdown fences:
{"symbol": "INFY", "alpha": 0.4, "reason": "Earnings beat reported 5 hours ago with raised FY27 guidance."}"""


def _build_user_message(symbol: str, sector: str, predicted_return: float,
                        indicator_snapshot: dict[str, float],
                        news_results: list[dict], regime_flag: int) -> str:
    parts = [
        f"symbol: {symbol}",
        f"sector: {sector}",
        f"predicted_return: {predicted_return:.4f}",
        f"regime_flag: {regime_flag}",
        "",
        "indicator_snapshot:",
        json.dumps(indicator_snapshot, indent=2),
        "",
        "news:",
    ]
    if news_results:
        for n in news_results:
            parts.append(
                f"- title: {n.get('title', '')}\n"
                f"  snippet: {n.get('snippet', '')}\n"
                f"  published_at: {n.get('published_at', '')}"
            )
    else:
        parts.append("(no news found)")

    return "\n".join(parts)


def judge_symbol(
    symbol: str,
    sector: str,
    predicted_return: float,
    indicator_snapshot: dict[str, float],
    news_results: list[dict],
    regime_flag: int,
    model_name: str = "deepseek-v4-flash",
    max_tokens: int = LLM_MAX_TOKENS,
    retry_attempts: int = LLM_RETRY_ATTEMPTS,
) -> tuple[float, str]:
    if not DEEPSEEK_API_KEY:
        logger.error("deepseek api key not configured")
        return (0.0, "api key not configured")

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    user_message = _build_user_message(
        symbol, sector, predicted_return, indicator_snapshot,
        news_results, regime_flag,
    )

    for attempt in range(1 + retry_attempts):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
        except Exception:
            logger.error("llm api call failed for %s (attempt %d/%d)",
                         symbol, attempt + 1, 1 + retry_attempts, exc_info=True)
            if attempt < retry_attempts:
                continue
            return (0.0, "api error")

        raw = response.choices[0].message.content or ""
        alpha, reason = parse_alpha(raw)
        return (alpha, reason)

    return (0.0, "api error")
