import logging
from pathlib import Path

import torch
import torch.nn as nn

from config import CHECKPOINT_DIR, TFT_HIDDEN_DIM

logger = logging.getLogger(__name__)


class PerStockAdapter(nn.Module):
    def __init__(self, hidden_dim: int = TFT_HIDDEN_DIM):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_state)


def adapter_path(sector: str, symbol: str,
                 checkpoint_dir: str = CHECKPOINT_DIR) -> Path:
    return Path(checkpoint_dir) / sector / f"{symbol}_adapter.pt"


def load_adapter(sector: str, symbol: str,
                 hidden_dim: int = TFT_HIDDEN_DIM,
                 checkpoint_dir: str = CHECKPOINT_DIR) -> PerStockAdapter:
    path = adapter_path(sector, symbol, checkpoint_dir)
    if path.exists():
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        adapter = PerStockAdapter(hidden_dim)
        adapter.load_state_dict(state)
        logger.info("loaded adapter for %s from %s", symbol, path)
        return adapter

    logger.info("no adapter found for %s at %s, returning zero-init", symbol, path)
    return PerStockAdapter(hidden_dim)


def save_adapter(adapter: PerStockAdapter, sector: str, symbol: str,
                 checkpoint_dir: str = CHECKPOINT_DIR) -> None:
    path = adapter_path(sector, symbol, checkpoint_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), str(path))
    logger.info("saved adapter for %s to %s", symbol, path)
