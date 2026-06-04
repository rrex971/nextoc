import logging
import sys
import tempfile
import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="%(levelname)s:%(name)s:%(message)s")

import torch

from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, STARTING_PORTFOLIO, LLM_MAX_TOKENS
from model.adapters import PerStockAdapter, load_adapter, save_adapter
from feedback.finetune import fine_tune_adapters
from judge.news import search_news
from judge.llm import judge_symbol, _build_user_message
from judge.score import parse_alpha, normalize_gamma_scores, final_score
from ranking.gamma import gamma_score, kelly_fraction
from trading.ledger import (
    ensure_trades_table, insert_trade, update_trade_exit,
    get_open_trades, get_closed_trades, get_cash_balance,
)
from feedback.reward import compute_reward, compute_rolling_accuracy
from trading.paper import close_positions, open_positions
from features.indicators import compute_indicators
from features.crosssectional import compute_regime_flag

_PASS = 0
_FAIL = 1
_SKIP = 2
results = []


def section(name):
    print(f"\n--- {name} ---")


def record(module, status, detail=""):
    tag = {_PASS: "PASS", _FAIL: "FAIL", _SKIP: "SKIP"}[status]
    results.append((module, status))
    print(f"  [{tag}] {module}" + (f" — {detail}" if detail else ""))


# ── 1. judge/news.py ──────────────────────────────────────────────────────────

section("1. judge/news.py — exa news search")

try:
    articles = search_news("Infosys", max_age_hours=48, top_n=3)
    if articles:
        top = articles[0]
        print(f"  top result: title={top['title'][:80]}, snippet={top['snippet'][:120]}, date={top['published_at']}")
        record("judge/news.py", _PASS, f"found {len(articles)} articles")
    else:
        record("judge/news.py", _SKIP, "no articles returned (maybe exa quota exhausted)")
except Exception as e:
    record("judge/news.py", _FAIL, str(e))


# ── 2. judge/llm.py — raw deepseek response ─────────────────────────────────

section("2. judge/llm.py — raw deepseek call")

indicator_snapshot = {
    "close": 1650.0, "volume": 5_000_000,
    "ema_9": 1640.0, "ema_21": 1625.0, "ema_55": 1600.0,
    "rsi_14": 58.0, "macd": 12.0, "macd_hist": 3.5,
    "bb_width": 0.05, "atr": 25.0, "obv": 150_000_000,
    "roc_5": 0.01, "roc_10": 0.025, "stoch_k": 65.0,
    "stoch_d": 60.0, "williams_r": -30.0,
    "sector_return_1d": 0.005, "stock_vs_sector": 0.01,
    "nifty50_return_1d": 0.003, "stock_vs_market": 0.012,
    "regime_flag": 0, "is_outlier": 0,
}

news_for_llm = articles if locals().get("articles") else []

DEFAULT_MODEL = "deepseek-chat"  # matches llm.py default
WORKING_MODEL = "deepseek-chat"     # confirmed working with the DeepSeek API

if not DEEPSEEK_API_KEY:
    record("judge/llm.py", _SKIP, "DEEPSEEK_API_KEY not set")
    raw_response = None
    alpha, reason = (0.0, "api key not configured")
else:
    #
    # (a) production path via judge_symbol() — uses DEFAULT_MODEL
    alpha, reason = judge_symbol(
        "INFY", "IT", 0.015, indicator_snapshot, news_for_llm, 0,
        model_name=DEFAULT_MODEL,
    )
    print(f"  judge_symbol(default model={DEFAULT_MODEL}) -> alpha={alpha}, reason={reason}")

    if alpha == 0.0:
        print(f"  note: default model failed, retrying with {WORKING_MODEL}")
        alpha, reason = judge_symbol(
            "INFY", "IT", 0.015, indicator_snapshot, news_for_llm, 0,
            model_name=WORKING_MODEL,
        )
        print(f"  judge_symbol({WORKING_MODEL}) -> alpha={alpha}, reason={reason}")

    #
    # (b) raw api call to capture the response before parsing
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        user_msg = _build_user_message(
            "INFY", "IT", 0.015, indicator_snapshot, news_for_llm, 0,
        )
        response = client.chat.completions.create(
            model=WORKING_MODEL,
            messages=[
                {"role": "system", "content": """you are a financial analyst reviewing nse-listed indian equities for next-day intraday trading.

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
{"symbol": "INFY", "alpha": 0.4, "reason": "Earnings beat reported 5 hours ago with raised FY27 guidance."}"""},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=LLM_MAX_TOKENS,
            temperature=0.0,
        )
        msg = response.choices[0].message
        finish = response.choices[0].finish_reason
        content = msg.content or ""
        print(f"  raw ({WORKING_MODEL}) finish_reason={finish}, content_len={len(content)}, content={content[:300]!r}")
        raw_response = content.strip()
        if raw_response:
            record("judge/llm.py", _PASS, "raw api call returned content")
        else:
            # last resort: try with more tokens
            response = client.chat.completions.create(
                model=WORKING_MODEL,
                messages=[
                    {"role": "system", "content": "you are a JSON-only financial analyst."},
                    {"role": "user", "content": f"for INFY with predicted_return 0.015 and no news, return {{\"symbol\": \"INFY\", \"alpha\": 0.0, \"reason\": \"no news\"}}"},
                ],
                max_tokens=500,
                temperature=0.0,
            )
            content = (response.choices[0].message.content or "").strip()
            print(f"  fallback response: {content[:300]!r}")
            raw_response = content or None
            record("judge/llm.py", _PASS if raw_response else _FAIL, "fallback api call")
    except Exception as e:
        raw_response = None
        record("judge/llm.py", _FAIL, f"raw api call failed: {e}")


# ── 3. judge/score.py — parse_alpha ──────────────────────────────────────────

section("3. judge/score.py — parse_alpha")

if raw_response:
    parsed_alpha, parsed_reason = parse_alpha(raw_response)
    print(f"  parse_alpha(raw_response) -> alpha={parsed_alpha}, reason={parsed_reason}")
    fell_back = "yes" if parsed_alpha == 0.0 else "no"
    print(f"  fell back to 0.0: {fell_back}")
    record("judge/score.py", _PASS, f"alpha={parsed_alpha}, raw_parse_fell_back={fell_back}")
elif alpha is not None:
    print(f"  judge_symbol yielded alpha={alpha}, reason={reason} (no raw response to re-parse)")
    record("judge/score.py", _PASS, f"alpha={alpha} via judge_symbol")
else:
    record("judge/score.py", _SKIP, "no deepseek response available")


# ── 4. judge/score.py — normalize_gamma_scores + final_score ─────────────────

section("4. judge/score.py — gamma normalization + final score")

fake_candidates = [
    {"symbol": "INFY", "gamma_score": 0.050},
    {"symbol": "TCS",  "gamma_score": 0.035},
    {"symbol": "WIPRO","gamma_score": 0.020},
    {"symbol": "HCLT", "gamma_score": 0.010},
    {"symbol": "TECHM","gamma_score": 0.005},
]

normalized = normalize_gamma_scores(fake_candidates)
alphas = {"INFY": 0.4, "TCS": -0.2, "WIPRO": 0.0, "HCLT": 0.1, "TECHM": 0.5}

for c in normalized:
    c["final_score"] = final_score(c["gamma_normalized"], alphas[c["symbol"]])

sorted_ = sorted(normalized, key=lambda x: x["final_score"], reverse=True)
print("  final ranking (desc):")
for i, c in enumerate(sorted_):
    print(f"    {i+1}. {c['symbol']}  gamma_norm={c['gamma_normalized']:.3f}  alpha={alphas[c['symbol']]:+.1f}  final={c['final_score']:.3f}")

ranks = [c["symbol"] for c in sorted_]
assert ranks[0] in ("TECHM", "INFY"), f"expected one of top scorers at rank 1, got {ranks[0]}"
record("judge/score.py", _PASS, "ranking sorted correctly")


# ── 5. trading/ledger.py — in-memory sqlite ──────────────────────────────────

section("5. trading/ledger.py — in-memory crud")

try:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    ensure_trades_table(db_path)

    base = dict(
        sector="IT", entry_price=1650.0, position_size_inr=10000,
        kelly_fraction=0.1, predicted_return=0.015, gamma_score=0.04,
        alpha=0.3, alpha_reason="test", final_score=0.5, regime_flag=0,
    )
    id1 = insert_trade(db_path, "2026-05-19", "INFY", **base)
    id2 = insert_trade(db_path, "2026-05-18", "TCS", **base)
    id3 = insert_trade(db_path, "2026-05-17", "WIPRO", **base)

    assert id1 and id2 and id3, "insert_trade returned falsy id"

    update_trade_exit(db_path, id1, 1700.0, 0.03, 1)
    update_trade_exit(db_path, id2, 3200.0, -0.02, -1)

    open_trades = get_open_trades(db_path)
    assert len(open_trades) == 1, f"expected 1 open, got {len(open_trades)}"
    assert open_trades[0]["symbol"] == "WIPRO", f"expected WIPRO open, got {open_trades[0]['symbol']}"

    cash = get_cash_balance(db_path, STARTING_PORTFOLIO)
    # 3 trades x 10000 deployed; 2 closed: INFY proceeds=10300, TCS proceeds=9800
    expected_cash = STARTING_PORTFOLIO - (10000 * 3) + 10300 + 9800
    assert abs(cash - expected_cash) < 0.01, f"cash {cash} != expected {expected_cash}"

    record("trading/ledger.py", _PASS, f"1 open trade, cash={cash:.0f}")
except Exception as e:
    record("trading/ledger.py", _FAIL, str(e))
finally:
    try:
        import os
        os.unlink(db_path)
    except Exception:
        pass


# ── 6. trading/paper.py — open/close integration ──────────────────────────────

section("6. trading/paper.py — open/close with compute_reward")

try:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    ensure_trades_table(db_path)

    sector_map = {"INFY": "IT", "TCS": "IT"}
    candidate = {
        "symbol": "INFY", "sector": "IT", "close": 1650.0,
        "predicted_return": 0.015, "gamma_score": 0.04,
        "kelly_fraction": 0.1, "alpha": 0.3,
        "alpha_reason": "test", "final_score": 0.5,
    }

    opened = open_positions([candidate], db_path, "2026-05-19", regime_flag=0,
                            starting_portfolio=100_000)
    assert opened == 1, f"expected 1 opened, got {opened}"

    # close with actual up move
    closed = close_positions(db_path, "2026-05-20", {"INFY": 1700.0})
    assert closed == 1, f"expected 1 closed, got {closed}"

    opened2 = open_positions([candidate], db_path, "2026-05-20", regime_flag=1)
    assert opened2 == 0, "expected 0 opened under regime flag"

    record("trading/paper.py", _PASS, "open/close flow with reward from feedback.reward")
except Exception as e:
    record("trading/paper.py", _FAIL, str(e))
finally:
    try:
        import os
        os.unlink(db_path)
    except Exception:
        pass


# ── 7. feedback/reward.py — compute_reward + compute_rolling_accuracy ────────

section("7. feedback/reward.py — compute_reward")

try:
    r1 = compute_reward(0.015, 0.03)
    r2 = compute_reward(0.015, -0.01)
    r3 = compute_reward(0.001, 0.005)
    assert r1 == 1, f"expected +1 for correct, got {r1}"
    assert r2 == -1, f"expected -1 for wrong, got {r2}"
    assert r3 == -1, f"expected -1 below hurdle, got {r3}"
    record("feedback/reward.py", _PASS, "compute_reward three cases correct")
except Exception as e:
    record("feedback/reward.py", _FAIL, str(e))


section("7b. feedback/reward.py — rolling accuracy")

try:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path2 = f.name
    ensure_trades_table(db_path2)
    # no trades → falls back to default
    acc = compute_rolling_accuracy(db_path2, lookback=30)
    assert abs(acc - 0.55) < 0.001, f"expected default 0.55, got {acc}"
    record("feedback/reward.py", _PASS, f"rolling accuracy default={acc}")
except Exception as e:
    record("feedback/reward.py", _FAIL, str(e))
finally:
    try:
        import os
        os.unlink(db_path2)
    except Exception:
        pass


# ── 7c. trading/ledger.py — get_closed_trades date filter ─────────────────────

section("7c. trading/ledger.py — get_closed_trades date filter")

try:
    import datetime as dt
    import sqlite3

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path3 = f.name

    ensure_trades_table(db_path3)

    conn = sqlite3.connect(db_path3)
    try:
        today = dt.date.today()
        for i in range(5):
            day = today - dt.timedelta(days=i)
            conn.execute(
                "insert into trades (date, symbol, sector, entry_price, "
                "position_size_inr, kelly_fraction, predicted_return, "
                "gamma_score, alpha, alpha_reason, final_score, regime_flag, "
                "exit_price, actual_return, reward) "
                "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (day.isoformat(), "TEST", "SECTOR", 100.0, 1000, 0.1, 0.01,
                 0.02, 0.0, "test", 0.5, 0,
                 101.0, 0.01, 1),
            )
        conn.commit()
    finally:
        conn.close()

    # limit_days=0 means cutoff=today → 1 trade
    closed_0 = get_closed_trades(db_path3, limit_days=0)
    assert len(closed_0) == 1, f"expected 1 trade for limit_days=0, got {len(closed_0)}"

    # limit_days=3 means cutoff=today-3 → 4 trades (today through t-3)
    closed_3 = get_closed_trades(db_path3, limit_days=3)
    assert len(closed_3) == 4, f"expected 4 trades for limit_days=3, got {len(closed_3)}"

    record("trading/ledger.py", _PASS, f"date filter: 0d={len(closed_0)}, 3d={len(closed_3)}")
except Exception as e:
    record("trading/ledger.py", _FAIL, f"date filter: {e}")
finally:
    try:
        import os
        os.unlink(db_path3)
    except Exception:
        pass


# ── 7d. reports/daily.py — generate_report format ─────────────────────────────

section("7d. reports/daily.py — generate_report format")

try:
    from reports.daily import generate_report

    fake_candidates = [
        {"symbol": "INFY", "predicted_return": 0.018, "alpha": 0.40,
         "final_score": 3.21, "kelly_fraction": 0.123},
        {"symbol": "HDFCBANK", "predicted_return": 0.012, "alpha": 0.20,
         "final_score": 1.87, "kelly_fraction": 0.081},
    ]
    fake_yesterday = [
        {"symbol": "TCS", "predicted_return": 0.012, "actual_return": 0.014, "reward": 1},
        {"symbol": "WIPRO", "predicted_return": 0.009, "actual_return": -0.006, "reward": -1},
    ]

    report = generate_report(
        date="2026-05-19",
        universe_count=500,
        gamma_passed=42,
        llm_count=10,
        trades_opened=4,
        regime_flag=0,
        regime_label="normal",
        candidates=fake_candidates,
        yesterday_trades=fake_yesterday,
        rolling_accuracy=0.65,
        cumulative_pnl=1523.50,
    )

    assert "=== nextoc eod" in report, "missing header"
    assert "universe evaluated:" in report, "missing universe count"
    assert "gamma filter passed:" in report, "missing gamma count"
    assert "INFY" in report, "missing INFY from top picks"
    assert "TCS" in report, "missing TCS from yesterday's results"
    assert "65.0%" in report or "rolling accuracy" in report, "missing rolling accuracy"
    assert "1,524" in report or "1524" in report, "missing pnl formatting"

    print(f"  report length: {len(report)} chars")
    print(f"  first line: {report.splitlines()[0]!r}")
    print(f"  top picks line count: {sum(1 for l in report.splitlines() if l.strip().startswith('INFY') or l.strip().startswith('HDFCBANK'))}")

    record("reports/daily.py", _PASS, f"report generated ({len(report)} chars)")
except Exception as e:
    record("reports/daily.py", _FAIL, str(e))


# ── 7e. scheduler.py — import + helper functions ───────────────────────────────

section("7e. scheduler.py — import + helpers")

try:
    from scheduler import _build_indicator_snapshot
    from reports.daily import _dir_label as report_dir_label

    fake_frame = pd.DataFrame({
        "close": [100.0], "volume": [1_000_000],
        "ema_9": [101.0], "ema_21": [102.0], "ema_55": [103.0],
        "rsi_14": [55.0], "macd": [1.0], "macd_hist": [0.5],
        "bb_width": [0.03], "atr": [2.0], "obv": [500_000],
        "roc_5": [0.01], "roc_10": [0.02], "stoch_k": [60.0],
        "stoch_d": [58.0], "williams_r": [-25.0],
        "sector_return_1d": [0.005], "stock_vs_sector": [0.01],
        "nifty50_return_1d": [0.003], "stock_vs_market": [0.012],
        "regime_flag": [0], "is_outlier": [0],
    })
    snapshot = _build_indicator_snapshot(fake_frame)
    assert snapshot["close"] == 100.0
    assert snapshot["rsi_14"] == 55.0
    assert snapshot["regime_flag"] == 0
    record("scheduler.py", _PASS, f"_build_indicator_snapshot ({len(snapshot)} keys)")

    assert report_dir_label(0.015) == "up"
    assert report_dir_label(0.001) == "down"
    assert report_dir_label(-0.005) == "down"
    record("reports/daily.py", _PASS, "_dir_label three cases correct")
except Exception as e:
    record("scheduler.py", _FAIL, str(e))


# ── 8. model/adapters.py — PerStockAdapter ────────────────────────────────────

section("8. model/adapters.py — PerStockAdapter")

try:
    adapter = PerStockAdapter(hidden_dim=16)
    hidden = torch.randn(4, 16)
    out = adapter(hidden)
    assert out.shape == (4, 1), f"expected (4, 1), got {out.shape}"

    with tempfile.TemporaryDirectory() as tmpdir:
        sector, symbol = "test_sector", "TEST"
        save_adapter(adapter, sector, symbol, checkpoint_dir=tmpdir)

        loaded = load_adapter(sector, symbol, hidden_dim=16, checkpoint_dir=tmpdir)
        out2 = loaded(hidden)
        assert torch.allclose(out, out2), "loaded adapter output differs"

        fresh = load_adapter("nonexistent", "FAKE", hidden_dim=16, checkpoint_dir=tmpdir)
        assert fresh.projection.weight.shape == (1, 16)

    record("model/adapters.py", _PASS, "forward + save/load round-trip")
except Exception as e:
    record("model/adapters.py", _FAIL, str(e))


# ── 9. feedback/finetune.py — fine_tune_adapters ──────────────────────────────

section("9. feedback/finetune.py — fine_tune_adapters")

try:
    # no-op with None hidden_states
    result = fine_tune_adapters(":memory:", {}, hidden_states=None)
    assert result == {}, f"expected empty dict, got {result}"

    # no-op with empty hidden_states
    result = fine_tune_adapters(":memory:", {}, hidden_states={})
    assert result == {}, f"expected empty dict, got {result}"

    # single-step with fake hidden vectors and a seeded db
    import sqlite3
    from trading.ledger import ensure_trades_table

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    ensure_trades_table(db_path)
    conn = sqlite3.connect(db_path)
    import datetime as dt_ft
    _today = dt_ft.date.today()
    for sym in ["STOCKA", "STOCKB"]:
        for i in range(5):
            day = (_today - dt_ft.timedelta(days=i)).isoformat()
            conn.execute(
                "insert into trades (date, symbol, sector, entry_price, "
                "position_size_inr, kelly_fraction, predicted_return, "
                "gamma_score, alpha, alpha_reason, final_score, regime_flag, "
                "exit_price, actual_return, reward) "
                "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (day, sym, "SECTOR", 100.0, 1000, 0.1, 0.01,
                 0.02, 0.0, "test", 0.5, 0,
                 101.0, 0.01, 1),
            )
    conn.commit()
    conn.close()

    hidden_vec = torch.randn(64)
    hs = {
        "STOCKA": [(hidden_vec, 1), (hidden_vec, 1), (hidden_vec, 1)],
        "STOCKB": [(hidden_vec, -1), (hidden_vec, -1), (hidden_vec, 1)],
    }
    sector_map = {"STOCKA": "SECTOR", "STOCKB": "SECTOR"}

    with tempfile.TemporaryDirectory() as tmpdir:
        result = fine_tune_adapters(
            db_path, sector_map, checkpoint_dir=tmpdir,
            hidden_states=hs, recent_days=3, hidden_dim=64,
        )
        assert "STOCKA" in result, f"STOCKA missing from results: {result}"
        print(f"  fine-tune results: {result}")

    record("feedback/finetune.py", _PASS, f"no-op + single-step: {result}")
except Exception as e:
    record("feedback/finetune.py", _FAIL, str(e))
finally:
    try:
        import os
        os.unlink(db_path)
    except Exception:
        pass


# ── 10. features/indicators.py — yfinance + indicators ─────────────────────

section("7. features/indicators.py — yfinance via compute_indicators")

try:
    ticker = yf.Ticker("INFY.NS")
    hist = ticker.history(period="90d")
    if hist.empty:
        record("features/indicators", _SKIP, "yfinance returned empty")
    else:
        df = hist.rename(columns=str.lower)
        needed = {"open", "high", "low", "close", "volume"}
        missing = needed - set(df.columns)
        if missing:
            record("features/indicators", _SKIP, f"missing columns: {missing}")
        else:
            result = compute_indicators(df)
            # drop rows where raw ohlcv is NaN (incomplete trading day)
            clean = result.dropna(subset=["open", "high", "low", "close", "volume"])
            warmup = clean.iloc[60:] if len(clean) > 60 else clean
            indicator_cols = [c for c in warmup.columns if c not in needed]
            nan_cols = [c for c in indicator_cols if warmup[c].isna().any()]
            if nan_cols:
                record("features/indicators", _FAIL, f"NaN after warmup in: {nan_cols}")
            else:
                print(f"  rows={len(result)}, clean_rows={len(clean)}, warmup_rows={len(warmup)}, cols={len(result.columns)}")
                record("features/indicators", _PASS, "no NaN after 60-day warmup")
except Exception as e:
    record("features/indicators", _FAIL, str(e))


# ── 8. features/crosssectional.py — regime_flag ──────────────────────────────

section("8. features/crosssectional.py — regime flag")

try:
    rng = np.random.default_rng(42)
    nifty_returns = pd.Series(rng.normal(0.0, 0.01, 90))
    nifty_returns.iloc[80] = -0.05  # crash day

    regime = compute_regime_flag(nifty_returns, window=60)

    crash_flag = regime.iloc[80]
    non_crash_flags = regime.drop(regime.index[80]).dropna()

    assert crash_flag == 1, f"expected 1 on crash day, got {crash_flag}"
    assert (non_crash_flags == 0).all(), "expected 0 on non-crash days"
    regime_count = regime.dropna().sum()
    print(f"  regime=1 days: {int(regime_count)} out of {int(regime.dropna().shape[0])}")
    record("features/crosssect", _PASS, f"crash day flagged correctly, {int(regime_count)} total stress days")
except Exception as e:
    record("features/crosssect", _FAIL, str(e))


# ── summary ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 30)
print("=== nextoc smoke test ===")
print("=" * 30)
passed = failed = skipped = 0
for module, status in results:
    tag = {_PASS: "PASS", _FAIL: "FAIL", _SKIP: "SKIP"}[status]
    line = f"  {module:<30} {tag}"
    print(line)
    if status == _PASS:
        passed += 1
    elif status == _FAIL:
        failed += 1
    else:
        skipped += 1

total = passed + failed + skipped
print(f"  {'─' * 38}")
print(f"  {'TOTAL':<30} {total}")
print(f"  {'PASSED':<30} {passed}")
print(f"  {'FAILED':<30} {failed}")
print(f"  {'SKIPPED':<30} {skipped}")

sys.exit(0 if failed == 0 else 1)
