import logging
import os

logger = logging.getLogger(__name__)


def _dir_label(value: float, hurdle: float = 0.002) -> str:
    return "up" if value > hurdle else "down"


def generate_report(
    date: str,
    universe_count: int,
    gamma_passed: int,
    llm_count: int,
    trades_opened: int,
    regime_flag: int,
    regime_label: str,
    candidates: list[dict],
    yesterday_trades: list[dict],
    rolling_accuracy: float,
    cumulative_pnl: float,
    log_file: str | None = None,
) -> str:
    lines = []
    lines.append(f"=== nextoc eod \u2014 {date} ===")
    lines.append(f"universe evaluated:  {universe_count} symbols")
    lines.append(f"gamma filter passed: {gamma_passed}")
    lines.append(f"llm judge run:       top {llm_count}")
    lines.append(f"trades placed:       {trades_opened}")
    lines.append(f"regime:              {regime_label}")
    lines.append("")

    lines.append("top picks:")
    lines.append(f"  {'symbol':<12} {'pred ret':>9} {'alpha':>7} {'final score':>12} {'kelly%':>7}")
    for c in candidates[:5]:
        pred_str = f"{c['predicted_return'] * 100:+.1f}%"
        alpha_str = f"{c.get('alpha', 0.0):+.2f}"
        fs_str = f"{c.get('final_score', 0.0):.2f}"
        kelly_str = f"{c.get('kelly_fraction', 0.0) * 100:.1f}%"
        lines.append(f"  {c['symbol']:<12} {pred_str:>9} {alpha_str:>7} {fs_str:>12} {kelly_str:>7}")

    lines.append("")
    lines.append("yesterday's results:")
    lines.append(f"  {'symbol':<12} {'pred dir':>9} {'actual dir':>11} {'reward':>7} {'actual ret':>11}")
    for t in yesterday_trades:
        pred_dir = _dir_label(t.get("predicted_return", 0))
        actual_dir = _dir_label(t.get("actual_return", 0))
        reward_str = f"{t.get('reward', 0):+d}"
        actual_ret_str = f"{t.get('actual_return', 0) * 100:+.1f}%"
        lines.append(f"  {t['symbol']:<12} {pred_dir:>9} {actual_dir:>11} {reward_str:>7} {actual_ret_str:>11}")

    lines.append("")
    lines.append(f"rolling accuracy (30d): {rolling_accuracy * 100:.1f}%")
    lines.append(f"paper p&l (cumulative): rs. {cumulative_pnl:,.0f}")

    report = "\n".join(lines)

    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            with open(log_file, "a") as f:
                f.write(report + "\n\n")
        except OSError:
            logger.error("failed to write report to %s", log_file)

    return report
