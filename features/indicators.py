import pandas as pd
import pandas_ta as ta


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema_9"] = ta.ema(df["close"], length=9)
    df["ema_21"] = ta.ema(df["close"], length=21)
    df["ema_55"] = ta.ema(df["close"], length=55)

    df["rsi_14"] = ta.rsi(df["close"], length=14)

    macd_df = ta.macd(df["close"])
    df["macd"] = macd_df.iloc[:, 0]
    df["macd_hist"] = macd_df.iloc[:, 1]

    bbands_df = ta.bbands(df["close"], length=20, std=2)
    # bb_width = (upper - lower) / middle
    df["bb_width"] = (
        bbands_df.iloc[:, 2] - bbands_df.iloc[:, 0]
    ) / bbands_df.iloc[:, 1]

    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    df["obv"] = ta.obv(df["close"], df["volume"])

    df["roc_5"] = ta.roc(df["close"], length=5)
    df["roc_10"] = ta.roc(df["close"], length=10)

    stoch_df = ta.stoch(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch_df.iloc[:, 0]
    df["stoch_d"] = stoch_df.iloc[:, 1]

    df["williams_r"] = ta.willr(df["high"], df["low"], df["close"], length=14)

    return df
