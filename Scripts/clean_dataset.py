"""
clean_dataset.py

Cleans the synthetic_raw.csv dataset and logs every change:
- Remove missing values
- Remove duplicate (date,ticker)
- Remove non-positive prices
- Filter out days with not exactly 50 tickers
- Sort by (date, signal desc)
- Enforce correct datatypes
- Save cleaned output: synthetic_clean.csv
- Print detailed logs of all modifications
"""

import pandas as pd
import numpy as np

INPUT_FILE = "data/synthetic_raw.csv"
OUTPUT_FILE = "data/synthetic_clean.csv"


def clean_dataset():

    print("\n====================================")
    print(" LOADING DATASET ")
    print("====================================")

    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    original_rows = len(df)

    print(f"Loaded {original_rows} rows\n")
    print("------------------------------------")

    change_log = {}

    # ---------------------------------------------------
    # 1. Remove missing values
    # ---------------------------------------------------
    missing_before = df.isna().sum().sum()
    df = df.dropna()
    missing_after = df.isna().sum().sum()

    removed_missing = missing_before - missing_after
    change_log["missing_values_removed"] = removed_missing

    print(f"Missing values removed: {removed_missing}")

    # ---------------------------------------------------
    # 2. Remove duplicates
    # ---------------------------------------------------
    dup_before = df.duplicated(subset=["date", "ticker"]).sum()
    df = df.drop_duplicates(subset=["date", "ticker"])
    dup_after = df.duplicated(subset=["date", "ticker"]).sum()

    removed_dupes = dup_before - dup_after
    change_log["duplicate_rows_removed"] = removed_dupes

    print(f"Duplicate (date,ticker) removed: {removed_dupes}")

    # ---------------------------------------------------
    # 3. Remove non-positive prices
    # ---------------------------------------------------
    bad_price_before = (df["close"] <= 0).sum()
    df = df[df["close"] > 0]
    bad_price_after = (df["close"] <= 0).sum()

    removed_bad_prices = bad_price_before - bad_price_after
    change_log["bad_prices_removed"] = removed_bad_prices

    print(f"Non-positive price rows removed: {removed_bad_prices}")

    # ---------------------------------------------------
    # 4. Ensure exactly 50 tickers per day
    # ---------------------------------------------------
    day_counts = df.groupby("date")["ticker"].nunique()
    invalid_days = day_counts[day_counts != 50].index.tolist()

    invalid_rows_removed = df[
        ~df["date"].isin(day_counts[day_counts == 50].index)
    ].shape[0]

    df = df[df["date"].isin(day_counts[day_counts == 50].index)]

    change_log["invalid_day_rows_removed"] = invalid_rows_removed

    print(
        f"Rows removed from days not having exactly 50 tickers: {invalid_rows_removed}"
    )

    # ---------------------------------------------------
    # 5. Sort by date then signal descending
    # ---------------------------------------------------
    df = df.sort_values(["date", "signal"], ascending=[True, False])
    change_log["sorted"] = True
    print("Sorted dataset by date and signal.")

    # ---------------------------------------------------
    # 6. Fix datatypes
    # ---------------------------------------------------
    df["ticker"] = df["ticker"].astype(str)
    df["close"] = df["close"].astype(float)
    df["signal"] = df["signal"].astype(float)
    change_log["datatype_fix"] = True

    print("Datatypes fixed (ticker=str, close=float, signal=float).")

    # ---------------------------------------------------
    # 7. Save cleaned dataset
    # ---------------------------------------------------
    df.to_csv(OUTPUT_FILE, index=False)

    final_rows = len(df)
    total_removed = original_rows - final_rows

    print("\n====================================")
    print(" CLEANING SUMMARY ")
    print("====================================")

    print(f"Original rows   : {original_rows}")
    print(f"Final rows      : {final_rows}")
    print(f"Total removed   : {total_removed}\n")

    for key, value in change_log.items():
        print(f"{key}: {value}")

    print("\n====================================")
    print(" CLEANING COMPLETE ")
    print(f" Saved cleaned dataset to: {OUTPUT_FILE}")
    print("====================================\n")

    if total_removed == 0:
        print("NOTE: Dataset was already perfectly clean. No changes made.")
    else:
        print("NOTE: Dataset required cleaning. Changes applied successfully.")


if __name__ == "__main__":
    clean_dataset()
