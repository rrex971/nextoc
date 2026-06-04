import logging

import torch
import torch.nn as nn

from config import ADAPTER_DEFAULT_LR, ADAPTER_FINETUNE_DAYS, ADAPTER_PENALTY_LR, CHECKPOINT_DIR, TFT_HIDDEN_DIM
from model.adapters import load_adapter, save_adapter
from trading.ledger import get_closed_trades

logger = logging.getLogger(__name__)


def fine_tune_adapters(
    db_path: str,
    sector_map: dict[str, str],
    checkpoint_dir: str = CHECKPOINT_DIR,
    hidden_states: dict[str, list[tuple[torch.Tensor, int]]] | None = None,
    recent_days: int = ADAPTER_FINETUNE_DAYS,
    hidden_dim: int = TFT_HIDDEN_DIM,
    lr_default: float = ADAPTER_DEFAULT_LR,
    lr_penalty: float = ADAPTER_PENALTY_LR,
) -> dict[str, int]:
    if not hidden_states:
        logger.info("no hidden states provided, skipping fine-tune")
        return {}

    closed = get_closed_trades(db_path, recent_days)
    symbol_trades: dict[str, list] = {}
    for trade in closed:
        symbol_trades.setdefault(trade["symbol"], []).append(trade)

    results: dict[str, int] = {}

    for symbol, trades in symbol_trades.items():
        if len(trades) < 1:
            logger.info("symbol %s has %d trades, need %d, skipping", symbol, len(trades), recent_days)
            continue

        if symbol not in hidden_states or not hidden_states[symbol]:
            logger.info("no hidden states for %s, skipping fine-tune", symbol)
            continue

        states_and_rewards = hidden_states[symbol]
        hidden_vectors = torch.stack([s for s, _ in states_and_rewards])
        rewards = torch.tensor([r for _, r in states_and_rewards], dtype=torch.float32)

        avg_reward = rewards.mean().item()
        lr = lr_penalty if avg_reward < 0 else lr_default

        sector = sector_map.get(symbol, "unknown")
        adapter = load_adapter(sector, symbol, hidden_dim, checkpoint_dir)
        adapter.train()

        targets = (rewards > 0).float().unsqueeze(1)
        logits = adapter(hidden_vectors)

        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)

        optim = torch.optim.SGD(adapter.parameters(), lr=lr)
        optim.zero_grad()
        loss.backward()
        optim.step()

        save_adapter(adapter, sector, symbol, checkpoint_dir)
        results[symbol] = len(trades)
        logger.info("fine-tuned adapter for %s: %d trades, avg_reward=%.3f, lr=%.1e", symbol, len(trades), avg_reward, lr)

    return results
