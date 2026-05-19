import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss

from config import TFT_HIDDEN_DIM

STATIC_CATEGORICALS = ["sector", "market_cap_tier"]
TIME_VARYING_KNOWN_REALS = ["day_of_week", "month"]
TIME_VARYING_UNKNOWN_REALS = [
    "close", "volume",
    "ema_9", "ema_21", "ema_55",
    "rsi_14", "macd", "macd_hist", "bb_width", "atr",
    "obv", "roc_5", "roc_10", "stoch_k", "stoch_d", "williams_r",
    "sector_return_1d", "stock_vs_sector",
    "nifty50_return_1d", "stock_vs_market",
    "regime_flag", "is_outlier",
]
TARGET = "return_1d_next"
MAX_ENCODER_LENGTH = 60
MAX_PREDICTION_LENGTH = 1


def _stack_frames(
    symbol_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for symbol, df in symbol_frames.items():
        frame = df.copy()
        frame["symbol"] = symbol
        frame = frame.reset_index()
        rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)
    combined["time_idx"] = combined.groupby("symbol").cumcount()
    combined[TARGET] = combined[TARGET].fillna(0.0)
    return combined


def build_dataset(
    symbol_frames: dict[str, pd.DataFrame],
    max_encoder_length: int = MAX_ENCODER_LENGTH,
    max_prediction_length: int = MAX_PREDICTION_LENGTH,
) -> tuple[TimeSeriesDataSet, pd.DataFrame]:
    combined = _stack_frames(symbol_frames)
    dataset = TimeSeriesDataSet(
        combined,
        time_idx="time_idx",
        target=TARGET,
        group_ids=["symbol"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=STATIC_CATEGORICALS,
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS + [TARGET],
    )
    return dataset, combined


def create_tft(
    dataset: TimeSeriesDataSet,
    hidden_size: int = TFT_HIDDEN_DIM,
    dropout: float = 0.1,
    attention_head_size: int = 4,
    hidden_continuous_size: int = 16,
    learning_rate: float = 1e-3,
) -> TemporalFusionTransformer:
    return TemporalFusionTransformer.from_dataset(
        dataset,
        hidden_size=hidden_size,
        dropout=dropout,
        attention_head_size=attention_head_size,
        hidden_continuous_size=hidden_continuous_size,
        learning_rate=learning_rate,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=4,
    )
