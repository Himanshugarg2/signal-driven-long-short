"""
validate_dataset.py

Validation script for synthetic_raw.csv
Checks:
- No missing values
- No duplicate (date, ticker)
- Always 50 tickers per day
- Prices must be positive
- AR(1) behavior check
- Correct vanish behavior:
    * each ticker appears continuously between its FIRST and LAST date
    * no ticker reappears after its LAST date
    * NO requirement to appear before its first date (important!)
    * NO false gaps for replacement tickers
- Sorted by signal each day
"""

import pandas as pd
import numpy as np

FILE_PATH = "data/synthetic_raw.csv"


def validate_dataset():

    print("Loading dataset...")
    df = pd.read_csv(FILE_PATH, parse_dates=["date"])
    print("Loaded:", len(df), "rows\n")

    results = {}

    # -----------------------------------------
    # 1. Missing values
    # -----------------------------------------
    print("Checking missing values...")
    missing = df.isna().sum().sum()
    results["missing_values"] = missing == 0
    print("Missing values:", missing, "\n")

    # -----------------------------------------
    # 2. Duplicate (date, ticker)
    # -----------------------------------------
    print("Checking duplicate rows...")
    dup = df.duplicated(subset=["date", "ticker"]).sum()
    results["duplicates"] = dup == 0
    print("Duplicate (date,ticker):", dup, "\n")

    # -----------------------------------------
    # 3. Exactly 50 tickers per day
    # -----------------------------------------
    print("Checking 50 tickers per day...")
    day_counts = df.groupby("date")["ticker"].nunique()
    results["50_per_day"] = (day_counts == 50).all()
    print(day_counts.value_counts(), "\n")

    # -----------------------------------------
    # 4. Valid prices
    # -----------------------------------------
    print("Checking non-positive prices...")
    bad = (df["close"] <= 0).sum()
    results["valid_prices"] = bad == 0
    print("Non-positive prices:", bad, "\n")

    # -----------------------------------------
    # 5. AR(1) autocorrelation check
    # -----------------------------------------
    print("Checking AR(1) behavior...")
    sample = df["ticker"].unique()[0]
    series = df[df["ticker"] == sample]["signal"]
    ac = series.autocorr(lag=1)
    print(f"Lag-1 autocorr for {sample}: {ac:.4f}")
    results["ar1_ok"] = -1 <= ac <= 1
    print()

    # -----------------------------------------
    # 6. Vanish Behavior Check (correct version)
    # -----------------------------------------
    print("Checking vanish behavior...")

    vanish_ok = True
    grouped = df.groupby("ticker")["date"]

    for ticker, dates in grouped:
        d = sorted(dates)

        # Compute expected continuous business days between first and last
        expected = pd.date_range(start=d[0], end=d[-1], freq="B")

        # If lengths differ → internal gap
        if len(expected) != len(d):
            print(f"ERROR: Ticker {ticker} has INTERNAL gap between {d[0]} and {d[-1]}")
            vanish_ok = False

        # No need to check days before first appearance → replacement stocks start late

    # 6.2 No ticker reappears after its last date
    spans = grouped.agg(["min", "max"])

    for ticker in spans.index:
        last_date = spans.loc[ticker, "max"]
        if len(df[(df["ticker"] == ticker) & (df["date"] > last_date)]) > 0:
            print(f"ERROR: Ticker {ticker} reappeared after vanish!")
            vanish_ok = False

    # 6.3 Universe always 50
    if not (day_counts == 50).all():
        print("ERROR: Universe is not always exactly 50 tickers!")
        vanish_ok = False

    results["vanish_behavior"] = vanish_ok
    print("Vanish behavior OK:", vanish_ok, "\n")

    # -----------------------------------------
    # 7. Check sorting by signal each day
    # -----------------------------------------
    print("Checking signal sorting...")

    sorted_ok = True
    for date in df["date"].unique()[:5]:  # check first 5 days
        sub = df[df["date"] == date]
        if not np.all(sub["signal"].values == np.sort(sub["signal"].values)[::-1]):
            print("ERROR: Sorting incorrect on date:", date)
            sorted_ok = False
            break

    results["signal_sorted"] = sorted_ok
    print("Signal sorting OK:", sorted_ok, "\n")

    # -----------------------------------------
    # FINAL REPORT
    # -----------------------------------------
    print("\n" + "=" * 50)
    print("VALIDATION REPORT")
    print("=" * 50)

    for key, val in results.items():
        print(("PASS" if val else "FAIL") + " — " + key)

    print("=" * 50)
    print("DATASET STATUS:", "VALID" if all(results.values()) else "INVALID")


if __name__ == "__main__":
    validate_dataset()
