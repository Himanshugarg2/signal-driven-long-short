"""
flag_dataset.py (FINAL CLEAN VERSION)

Flags all tickers that will disappear within the next H trading days.

Adds:
- days_to_vanish_trading : number of future TRADING DAYS before ticker disappears
- disappears_t1 : disappears tomorrow → MUST EXIT today
- unsafe_to_trade : True if ticker disappears in next H days (H = holding_period)

Input:  data/synthetic_clean.csv
Output: data/synthetic_flagged.csv
"""

import pandas as pd

INPUT_FILE = "data/synthetic_clean.csv"
OUTPUT_FILE = "data/synthetic_flagged.csv"


HOLDING_PERIOD = 3  # Change it according to backtest config


def flag_dataset():

    print("\n==============================")
    print(" LOADING CLEANED DATASET ")
    print("==============================")

    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    print("Loaded:", len(df), "rows")

    # Sort properly
    df = df.sort_values(["ticker", "date"]).copy()

    df["rank_desc"] = (
        df.groupby("ticker")["date"].rank(method="first", ascending=False).astype(int)
    )

    df["days_to_vanish_trading"] = df["rank_desc"] - 1

    # If ticker disappears TOMORROW → must exit today
    df["disappears_t1"] = df["days_to_vanish_trading"] == 1

    # Unsafe to trade if disappearing within holding period
    df["unsafe_to_trade"] = df["days_to_vanish_trading"].between(1, HOLDING_PERIOD)

    print("\nFlag counts:")
    print(df[["disappears_t1", "unsafe_to_trade"]].sum())

    print("\nSample vanish cases:\n")
    sample = df[df["unsafe_to_trade"] == True].head(10)
    print(
        sample[
            [
                "ticker",
                "date",
                "days_to_vanish_trading",
                "disappears_t1",
                "unsafe_to_trade",
            ]
        ]
    )

    # Cleanup helper
    df.drop(columns=["rank_desc"], inplace=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n==============================")
    print(" FLAGGING COMPLETE ")
    print("==============================")
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    flag_dataset()
