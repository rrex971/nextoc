import logging
from pathlib import Path

import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

from config import CHECKPOINT_DIR
from model.tft import build_dataset, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH

logger = logging.getLogger(__name__)


def predict_returns(
    symbol_frames: dict[str, pd.DataFrame],
    sector_map: dict[str, str],
    checkpoint_dir: str = CHECKPOINT_DIR,
    batch_size: int = 64,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    sectors: dict[str, list[str]] = {}
    for symbol, sector in sector_map.items():
        if symbol in symbol_frames:
            sectors.setdefault(sector, []).append(symbol)

    results: dict[str, float] = {}
    hidden_states: dict[str, torch.Tensor] = {}

    for sector, symbols in sectors.items():
        ckpt_path = Path(checkpoint_dir) / sector / "tft.ckpt"
        if not ckpt_path.exists():
            logger.warning("no checkpoint for sector %s", sector)
            continue

        sector_data = {s: symbol_frames[s] for s in symbols}
        min_rows = MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH
        truncated = {}
        for sym, df in sector_data.items():
            if len(df) < min_rows:
                logger.warning("symbol %s has only %d days, need %d", sym, len(df), min_rows)
                continue
            truncated[sym] = df.tail(min_rows)

        if not truncated:
            continue

        dataset, data = build_dataset(truncated)
        predict_dataset = TimeSeriesDataSet.from_dataset(
            dataset, data, predict=True
        )
        dataloader = predict_dataset.to_dataloader(
            train=False, batch_size=batch_size, num_workers=0
        )

        tft = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_path))
        tft.eval()

        hidden_capture: list[torch.Tensor] = []
        def _capture_hook(module, input, output):
            hidden_capture.append(input[0].detach().cpu())

        handle = tft.output_layer.register_forward_hook(_capture_hook)

        result = tft.predict(
            dataloader,
            mode="prediction",
            return_index=True,
        )

        handle.remove()

        offset = 0
        for batch_idx, batch_hidden in enumerate(hidden_capture):
            batch_size_actual = batch_hidden.shape[0]
            for i in range(batch_size_actual):
                sym = result.index.iloc[offset + i]["symbol"]
                hidden_states[sym] = batch_hidden[i, 0, :]
            offset += batch_size_actual

        for i in range(len(result.output)):
            symbol = result.index.iloc[i]["symbol"]
            results[symbol] = result.output[i, 0].item()

    assert len(hidden_states) == len(results), (
        f"hidden state alignment mismatch: {len(hidden_states)} states for {len(results)} symbols"
    )
    for sym in results:
        if sym not in hidden_states:
            logger.warning("symbol %s missing from hidden states", sym)

    return results, hidden_states
