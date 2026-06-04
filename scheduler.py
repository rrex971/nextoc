import datetime as dt
import json
import logging
import os
import sys

import pandas as pd
import yfinance as yf

from config import (
    CHECKPOINT_DIR, DB_PATH, LOG_DIR, LOG_FILE,
    HURDLE, LIQUIDITY_FLOOR, TOP_K,
    KELLY_MULTIPLIER, MAX_POSITION_PCT, WIN_PROBABILITY_DEFAULT,
    LAMBDA, NEWS_MAX_AGE_HOURS, NEWS_TOP_N,
    STARTING_PORTFOLIO, MIN_LOOKBACK_DAYS,
    ADAPTER_FINETUNE_DAYS,
)
from data.calendar import is_trading_day
from data.fetch import fetch_ohlcv
from data.universe import load_universe
from features.pipeline import build_features
from features.crosssectional import compute_regime_flag
from model.inference import predict_returns
from ranking.gamma import gamma_score, kelly_fraction, rank_by_gamma
from judge.news import search_news
from judge.llm import judge_symbol
from judge.score import (
    normalize_gamma_scores, final_score, parse_alpha,
    ensure_llm_triples_table, write_llm_triple,
)
from trading.ledger import (
    ensure_trades_table, get_open_trades, get_closed_trades,
    get_cash_balance,
)
from trading.paper import close_positions, open_positions
from feedback.finetune import fine_tune_adapters
from feedback.reward import compute_rolling_accuracy
from reports.daily import generate_report

logger = logging.getLogger(__name__)

_logging_initialized = False


def _build_indicator_snapshot(frame: pd.DataFrame) -> dict[str, float]:
    latest = frame.iloc[-1]
    keys = [
        "close", "volume",
        "ema_9", "ema_21", "ema_55",
        "rsi_14", "macd", "macd_hist", "bb_width", "atr",
        "obv", "roc_5", "roc_10", "stoch_k", "stoch_d", "williams_r",
        "sector_return_1d", "stock_vs_sector",
        "nifty50_return_1d", "stock_vs_market",
        "regime_flag", "is_outlier",
    ]
    return {k: float(latest.get(k, 0.0)) for k in keys}


def _fetch_nifty_return(from_date: str, to_date: str) -> pd.Series | None:
    try:
        df = fetch_ohlcv("NIFTY50", from_date, to_date)
        if df is not None and not df.empty:
            returns = df.set_index("timestamp")["close"].pct_change().dropna()
            logger.info("fetched nifty return via groww (%d days)", len(returns))
            return returns
    except Exception:
        logger.warning("groww nifty fetch failed, falling back to yfinance")

    try:
        ticker = yf.Ticker("^NSEI")
        hist = ticker.history(start=from_date, end=to_date)
        if hist.empty:
            logger.error("yfinance returned empty for ^NSEI")
            return None
        returns = hist["Close"].pct_change().dropna()
        returns.index = pd.to_datetime(returns.index)
        logger.info("fetched nifty return via yfinance (%d days)", len(returns))
        return returns
    except Exception:
        logger.error("yfinance nifty fetch also failed")
        return None


def _setup_logging(log_dir: str = LOG_DIR, log_file: str = LOG_FILE) -> None:
    global _logging_initialized
    if _logging_initialized:
        return
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)
    _logging_initialized = True


def run_pipeline(date_str: str | None = None, dry_run: bool = False) -> bool:
    if date_str:
        today = dt.date.fromisoformat(date_str)
    else:
        today = dt.date.today()

    yesterday = today - dt.timedelta(days=1)
    date_str = today.isoformat()
    yesterday_str = yesterday.isoformat()

    _setup_logging()
    dir = os.path.dirname(DB_PATH) or "."
    os.makedirs(dir, exist_ok=True)
    ensure_trades_table(DB_PATH)
    ensure_llm_triples_table(DB_PATH)

    if dry_run:
        logger.info("DRY RUN — no trades will be recorded")

    logger.info("pipeline run starting for %s", date_str)

    # 1. pre-check
    if not is_trading_day(today):
        logger.info("%s is not a trading day, exiting", date_str)
        return False

    logger.info("market check passed for %s", date_str)

    # 2. close yesterday's positions
    if dry_run:
        logger.info("DRY RUN: would close positions")
    else:
        try:
            close_prices: dict[str, float] = {}
            open_trades = get_open_trades(DB_PATH)
            if open_trades:
                for trade in open_trades:
                    try:
                        df = fetch_ohlcv(trade["symbol"], date_str, date_str)
                        if df is not None and not df.empty:
                            close_prices[trade["symbol"]] = float(df.iloc[-1]["close"])
                    except Exception:
                        logger.warning("failed to fetch close for %s, skipping close", trade["symbol"])

                closed = close_positions(DB_PATH, date_str, close_prices)
                logger.info("closed %d positions", closed)
            else:
                logger.info("no open positions to close")
        except Exception:
            logger.exception("error closing positions")

    # 4. fetch universe + ohlcv
    try:
        universe = load_universe()
        logger.info("loaded %d universe entries", len(universe))
    except Exception:
        logger.exception("failed to load universe")
        return False

    sector_map: dict[str, str] = {}
    cap_tier_map: dict[str, str] = {}
    company_name_map: dict[str, str] = {}
    for entry in universe:
        sym = entry["symbol"]
        sector_map[sym] = entry.get("sector", "")
        cap_tier_map[sym] = entry.get("market_cap_tier", "")
        company_name_map[sym] = entry.get("company_name", sym)

    # fetch training data window: last 365 days for enough training history
    from_date_obj = today - dt.timedelta(days=365)
    from_date = from_date_obj.isoformat()

    raw_frames: dict[str, pd.DataFrame] = {}
    for entry in universe:
        sym = entry["symbol"]
        try:
            df = fetch_ohlcv(sym, from_date, date_str)
            raw_frames[sym] = df
        except Exception:
            logger.warning("failed to fetch %s, skipping", sym)

    logger.info("fetched ohlcv for %d / %d symbols", len(raw_frames), len(universe))

    # 5. features
    try:
        nifty_return = _fetch_nifty_return(from_date, date_str)
        if nifty_return is None or nifty_return.empty:
            logger.error("nifty return unavailable, cannot compute features")
            return False
    except Exception:
        logger.exception("failed to fetch nifty return")
        return False

    try:
        feature_frames = build_features(raw_frames, sector_map, cap_tier_map, nifty_return)
        logger.info("built features for %d symbols", len(feature_frames))
    except Exception:
        logger.exception("feature building failed")
        return False

    # extract regime_flag from the latest common day
    regime_flag = 0
    if feature_frames:
        sample_sym = next(iter(feature_frames))
        latest_flag = feature_frames[sample_sym].iloc[-1].get("regime_flag", 0)
        regime_flag = int(latest_flag)

    universe_count = len(set(feature_frames.keys()) & set(raw_frames.keys()))
    gamma_passed = 0
    llm_count = 0
    trades_opened = 0

    # 6. predict + hidden states
    hidden_states: dict = {}
    try:
        predictions, hidden_states = predict_returns(feature_frames, sector_map, CHECKPOINT_DIR)
        logger.info("predicted returns for %d symbols", len(predictions))
    except Exception:
        logger.exception("prediction failed")
        predictions = {}

    # 7. fine-tune with hidden states
    if dry_run:
        logger.info("DRY RUN: skipping fine-tune")
    else:
        try:
            closed_trades = get_closed_trades(DB_PATH, recent_days=ADAPTER_FINETUNE_DAYS)
            symbol_rewards: dict[str, list[dict]] = {}
            for trade in closed_trades:
                symbol_rewards.setdefault(trade["symbol"], []).append(trade)

            ft_hidden_states: dict[str, list] = {}
            for symbol, trades in symbol_rewards.items():
                if symbol in hidden_states and trades:
                    ft_hidden_states[symbol] = [
                        (hidden_states[symbol], t["reward"])
                        for t in trades
                    ]

            if ft_hidden_states:
                ft_results = fine_tune_adapters(
                    DB_PATH, sector_map, CHECKPOINT_DIR,
                    hidden_states=ft_hidden_states,
                )
                logger.info("fine-tune results: %s", ft_results)
            else:
                logger.info("no symbols with both hidden states and closed trades, skipping fine-tune")
        except Exception:
            logger.exception("error during fine-tune")

    # 8. rank
    try:
        candidates = rank_by_gamma(predictions, feature_frames)
        gamma_passed = len(candidates)
        logger.info("gamma filter: %d candidates passed", gamma_passed)
    except Exception:
        logger.exception("gamma ranking failed")
        candidates = []

    scored = []
    if not candidates:
        logger.info("no candidates passed gamma filter, skipping judge and open")
    else:
        # 9. judge
        scored = normalize_gamma_scores(candidates)
        for c in scored:
            company_name = company_name_map.get(c["symbol"], c["symbol"])
            frame = feature_frames.get(c["symbol"])
            if frame is None or frame.empty:
                c["alpha"] = 0.0
                c["alpha_reason"] = "no feature data"
                c["final_score"] = final_score(c["gamma_normalized"], 0.0)
                llm_count += 1
                continue

            snapshot = _build_indicator_snapshot(frame)

            try:
                news = search_news(company_name, NEWS_MAX_AGE_HOURS, NEWS_TOP_N)
            except Exception:
                logger.warning("news search failed for %s", c["symbol"])
                news = []

            news_json = json.dumps(news)

            try:
                alpha, reason = judge_symbol(
                    c["symbol"], sector_map.get(c["symbol"], ""),
                    c["predicted_return"], snapshot, news, regime_flag,
                )
            except Exception:
                logger.warning("llm judge failed for %s, alpha=0.0", c["symbol"])
                alpha, reason = 0.0, "llm error"

            c["alpha"] = alpha
            c["alpha_reason"] = reason
            c["final_score"] = final_score(c["gamma_normalized"], alpha)

            write_llm_triple(DB_PATH, date_str, c["symbol"], news_json, alpha, reason)
            llm_count += 1

        # 10. open trades
        scored.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        if dry_run:
            logger.info("DRY RUN: would open top %d positions", len(scored))
        else:
            try:
                trades_opened = open_positions(scored, DB_PATH, date_str, regime_flag)
                logger.info("opened %d positions", trades_opened)
            except Exception:
                logger.exception("failed to open positions")

    # 11. report
    try:
        closed = get_closed_trades(DB_PATH, limit_days=1)
        yesterday_trades = [t for t in closed if t.get("date", "") == yesterday_str]

        rolling_acc = compute_rolling_accuracy(DB_PATH, lookback=30)

        cash = get_cash_balance(DB_PATH, STARTING_PORTFOLIO)
        open_value = sum(t["position_size_inr"] for t in get_open_trades(DB_PATH))
        cumulative_pnl = cash + open_value - STARTING_PORTFOLIO

        regime_label = "STRESS" if regime_flag else "normal"

        report = generate_report(
            date=date_str,
            universe_count=universe_count,
            gamma_passed=gamma_passed,
            llm_count=llm_count,
            trades_opened=trades_opened,
            regime_flag=regime_flag,
            regime_label=regime_label,
            candidates=scored if candidates else [],
            yesterday_trades=yesterday_trades,
            rolling_accuracy=rolling_acc,
            cumulative_pnl=cumulative_pnl,
            log_file=None if dry_run else LOG_FILE,
        )
        if dry_run:
            print(f"\n(DRY RUN) {report}\n")
        else:
            print("\n" + report + "\n")
    except Exception:
        logger.exception("failed to generate report")

    logger.info("pipeline run complete for %s", date_str)
    return True


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    date_arg = next((a for a in sys.argv[1:] if not a.startswith("--")), None)
    success = run_pipeline(date_arg, dry_run=dry_run)
    sys.exit(0 if success else 1)
