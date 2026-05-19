import logging
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_forecasting import TimeSeriesDataSet

from config import CHECKPOINT_DIR, TRAINING_LOOKBACK_DAYS
from model.tft import build_dataset, create_tft, MAX_ENCODER_LENGTH

logger = logging.getLogger(__name__)


def train_sector_model(
    sector: str,
    symbol_frames: dict[str, pd.DataFrame],
    checkpoint_dir: str = CHECKPOINT_DIR,
    max_epochs: int = 30,
    batch_size: int = 64,
) -> str:
    if not symbol_frames:
        raise ValueError(f"no symbols for sector {sector}")

    logger.info("training tft for sector %s (%d symbols)", sector, len(symbol_frames))

    # keep only the last TRAINING_LOOKBACK_DAYS per symbol
    trimmed = {
        sym: df.tail(TRAINING_LOOKBACK_DAYS)
        for sym, df in symbol_frames.items()
        if len(df) >= TRAINING_LOOKBACK_DAYS
    }
    if not trimmed:
        raise ValueError(f"no symbols with >= {TRAINING_LOOKBACK_DAYS} days in sector {sector}")

    dataset, data = build_dataset(trimmed)

    max_t = data["time_idx"].max()
    cutoff = max(0, int(max_t * 0.8))

    training = TimeSeriesDataSet.from_dataset(
        dataset,
        data[data.time_idx < cutoff],
        stop_randomization=True,
    )
    validation = TimeSeriesDataSet.from_dataset(
        dataset,
        data,
        stop_randomization=True,
        min_prediction_idx=max(cutoff, max_t - 60),
    )

    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    tft = create_tft(training)

    ckpt_path = Path(checkpoint_dir) / sector
    ckpt_path.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        ModelCheckpoint(
            dirpath=str(ckpt_path),
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        gradient_clip_val=0.1,
        callbacks=callbacks,
        logger=CSVLogger(str(ckpt_path), name="logs"),
        enable_progress_bar=False,
    )

    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    final_path = ckpt_path / "tft.ckpt"
    trainer.save_checkpoint(str(final_path))
    logger.info("sector %s model saved to %s", sector, final_path)
    return str(final_path)


def train_all_sectors(
    symbol_frames: dict[str, pd.DataFrame],
    sector_map: dict[str, str],
) -> dict[str, str]:
    sectors: dict[str, list[str]] = {}
    for symbol, sector in sector_map.items():
        if symbol in symbol_frames:
            sectors.setdefault(sector, []).append(symbol)

    checkpoints: dict[str, str] = {}
    for sector, symbols in sectors.items():
        sector_data = {s: symbol_frames[s] for s in symbols}
        try:
            path = train_sector_model(sector, sector_data)
            checkpoints[sector] = path
        except Exception:
            logger.exception("failed to train sector %s", sector)
    return checkpoints
