import pandas as pd
import numpy as np
import os

INPUT_FILE = "data/synthetic_flagged.csv"
os.makedirs("data/trading", exist_ok=True)
OUT_PRICES = "data/trading/trading_prices.csv"
OUT_WEIGHTS = "data/trading/trading_weights.csv"

# Weights
LONG_W = 0.80 / 5
SHORT_W = -0.20 / 5


def build_trading_dataset():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    df = df.sort_values(["date", "signal"], ascending=[True, False])

    prices = df.pivot(index="date", columns="ticker", values="close")
    signals = df.pivot(index="date", columns="ticker", values="signal")
    unsafe = df.pivot(index="date", columns="ticker", values="unsafe_to_trade").fillna(
        True
    )

    # Initialize with NaN
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    dates = prices.index.to_list()

    print("Building Weights (Strict 4-Day Check)...")

    for i, date in enumerate(dates):

        # REBALANCE DAY
        if i % 3 == 0:

            # --- FIX 1: Explicitly reset this day to 0.0 ---
            # This ensures that stocks NOT selected today get a target weight of 0.0 (Sell)
            weights.loc[date, :] = 0.0

            today_signals = signals.loc[date].dropna()
            today_unsafe = unsafe.loc[date].loc[today_signals.index]
            tradable = today_signals[today_unsafe == False]

            # Strict 4-Day Price Check
            if i + 3 < len(dates):
                d0 = prices.loc[dates[i]]
                d1 = prices.loc[dates[i + 1]]
                d2 = prices.loc[dates[i + 2]]
                d3 = prices.loc[dates[i + 3]]

                valid_4day = tradable.index[
                    d0[tradable.index].notna()
                    & d1[tradable.index].notna()
                    & d2[tradable.index].notna()
                    & d3[tradable.index].notna()
                ]
                tradable = tradable.loc[valid_4day]
            else:
                tradable = tradable.iloc[0:0]

            longs = tradable.sort_values(ascending=False).head(5).index
            shorts = tradable.sort_values(ascending=True).head(5).index

            for t in longs:
                weights.loc[date, t] = LONG_W
            for t in shorts:
                weights.loc[date, t] = SHORT_W

    prices.to_csv(OUT_PRICES)
    weights.to_csv(OUT_WEIGHTS)
    print("Done.")


if __name__ == "__main__":
    build_trading_dataset()
