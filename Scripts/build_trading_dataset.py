import pandas as pd
import numpy as np
import os

INPUT_FILE = "data/synthetic_flagged.csv"
os.makedirs("data/trading", exist_ok=True)
OUT_PRICES = "data/trading/trading_prices.csv"
OUT_WEIGHTS = "data/trading/trading_weights.csv"


HOLDING_PERIOD = 3  # Rebalance every N days
N_LONGS = 5  # Number of Long positions
N_SHORTS = 5  # Number of Short positions

LONG_ALLOCATION = 0.80  # Total capital for Longs (80%)
SHORT_ALLOCATION = -0.20  # Total capital for Shorts (20%)

LONG_W = LONG_ALLOCATION / N_LONGS
SHORT_W = SHORT_ALLOCATION / N_SHORTS


def build_trading_dataset():
    print(
        f"Loading data... (Config: Hold {HOLDING_PERIOD} days, {N_LONGS} Longs, {N_SHORTS} Shorts)"
    )

    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    df = df.sort_values(["date", "signal"], ascending=[True, False])

    prices = df.pivot(index="date", columns="ticker", values="close")
    signals = df.pivot(index="date", columns="ticker", values="signal")
    unsafe = df.pivot(index="date", columns="ticker", values="unsafe_to_trade").fillna(
        True
    )

    # Initialize weights matrix with NaN
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    dates = prices.index.to_list()

    print("Building Weights with Dynamic Checks...")

    for i, date in enumerate(dates):

        if i % HOLDING_PERIOD == 0:

            # Explicitly reset this day to 0.0 (Sell unselected stocks)
            weights.loc[date, :] = 0.0

            today_signals = signals.loc[date].dropna()
            today_unsafe = unsafe.loc[date].loc[today_signals.index]
            tradable = today_signals[today_unsafe == False]

            # Check if we have enough future data left
            if i + HOLDING_PERIOD < len(dates):

                valid_mask = pd.Series(True, index=tradable.index)

                for offset in range(HOLDING_PERIOD + 1):
                    future_date = dates[i + offset]
                    price_series = prices.loc[future_date]
                    valid_mask = valid_mask & price_series[tradable.index].notna()

                # Filter tradable list using the final mask
                tradable = tradable.loc[valid_mask[valid_mask].index]

            else:
                tradable = tradable.iloc[0:0]

            longs = tradable.sort_values(ascending=False).head(N_LONGS).index
            shorts = tradable.sort_values(ascending=True).head(N_SHORTS).index

            for t in longs:
                weights.loc[date, t] = LONG_W
            for t in shorts:
                weights.loc[date, t] = SHORT_W

    prices.to_csv(OUT_PRICES)
    weights.to_csv(OUT_WEIGHTS)
    print("Done. Dataset built successfully.")


if __name__ == "__main__":
    build_trading_dataset()
