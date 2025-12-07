"""
flag_dataset.py (CORRECTED)

Flags all tickers that will disappear within the next 1, 2, or 3 TRADING days.

Adds:
- days_to_vanish_trading : number of future trading days before ticker disappears
- disappears_t1 : disappears tomorrow → must EXIT today
- disappears_t2 : disappears in 2 trading days → cannot BUY today
- disappears_t3 : disappears in 3 trading days → cannot BUY today
- unsafe_to_trade : True if disappears in next 1–3 trading days → DO NOT BUY

Input:  data/synthetic_clean.csv
Output: data/synthetic_flagged.csv
"""

import pandas as pd
import numpy as np

INPUT_FILE = "data/synthetic_clean.csv"
OUTPUT_FILE = "data/synthetic_flagged.csv"


def flag_dataset():

    print("\n==============================")
    print(" LOADING CLEANED DATASET ")
    print("==============================")

    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    print("Loaded:", len(df), "rows")

    # Ensure proper ordering
    df = df.sort_values(["ticker", "date"]).copy()

    df["rank_desc"] = (
        df.groupby("ticker")["date"].rank(method="first", ascending=False).astype(int)
    )
    df["days_to_vanish_trading"] = df["rank_desc"] - 1
    df["disappears_t1"] = df["days_to_vanish_trading"] == 1
    df["disappears_t2"] = df["days_to_vanish_trading"] == 2
    df["disappears_t3"] = df["days_to_vanish_trading"] == 3
    df["unsafe_to_trade"] = df["days_to_vanish_trading"].between(1, 3)
    print("\nFlag counts:")
    print(
        df[["disappears_t1", "disappears_t2", "disappears_t3", "unsafe_to_trade"]].sum()
    )

    sample = df[df["days_to_vanish_trading"].between(1, 3)].head(10)

    print("\nSample vanish cases:\n")
    print(
        sample[
            [
                "ticker",
                "date",
                "days_to_vanish_trading",
                "disappears_t1",
                "disappears_t2",
                "disappears_t3",
            ]
        ]
    )

    # Cleanup helper column
    df.drop(columns=["rank_desc"], inplace=True)

    # ---------------------------------------------------------
    # Save output
    # ---------------------------------------------------------
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n==============================")
    print(" FLAGGING COMPLETE ")
    print("==============================")
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    flag_dataset()
