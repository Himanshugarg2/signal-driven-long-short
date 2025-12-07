import pandas as pd
import numpy as np
import vectorbt as vbt
import os

PRICES_FILE = "data/trading/trading_prices.csv"
WEIGHTS_FILE = "data/trading/trading_weights.csv"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# Must match the period used in build_trading_dataset.py
HOLDING_PERIOD = 3


def run_backtest():

    print("\n==============================")
    print(" RUNNING VECTORBT BACKTEST ")
    print("==============================\n")

    # 1. Load data
    prices = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
    weights = pd.read_csv(WEIGHTS_FILE, index_col=0, parse_dates=True)

    print("Prices shape :", prices.shape)
    print("Weights shape:", weights.shape)

    # 2. Run Portfolio (NaN = Hold, 0.0 = Close)
    portfolio = vbt.Portfolio.from_orders(
        close=prices,
        size=weights,
        size_type="targetpercent",
        init_cash=1_000_000,
        fees=0.0012,
        cash_sharing=True,
        freq="1D",
        call_seq="auto",
    )

    # ----------------------------------------------------------
    # STATS & VALIDATION
    # ----------------------------------------------------------
    print("\n==============================")
    print(" BACKTEST COMPLETE ")
    print("==============================\n")

    stats = portfolio.stats()
    print(stats)

    stats.to_csv(f"{RESULT_DIR}/backtest_stats.csv")
    portfolio.value().to_csv(f"{RESULT_DIR}/equity_curve.csv")

    # ==========================================================
    # TRADE LOG & DURATION CHECK
    # ==========================================================
    trade_records = portfolio.trades.records

    trade_df = pd.DataFrame(
        {
            "col": trade_records["col"],
            "entry_idx": trade_records["entry_idx"],
            "exit_idx": trade_records["exit_idx"],
        }
    )

    # Map indices to readable dates/tickers
    date_list = prices.index.to_list()
    ticker_list = prices.columns.to_list()

    trade_df["ticker"] = trade_df["col"].apply(lambda i: ticker_list[i])
    trade_df["entry_date"] = trade_df["entry_idx"].apply(lambda i: date_list[i])
    trade_df["exit_date"] = trade_df["exit_idx"].apply(lambda i: date_list[i])

    # Calculate Duration
    trade_df["duration_days"] = (trade_df["exit_date"] - trade_df["entry_date"]).dt.days

    trade_df.to_csv(f"{RESULT_DIR}/trade_log.csv", index=False)

    # Check for Short Trades based on dynamic HOLDING_PERIOD
    total = len(trade_df)
    short = len(trade_df[trade_df["duration_days"] < HOLDING_PERIOD])

    print("\n==============================")
    print(" TRADE DURATION SUMMARY ")
    print("==============================")
    print(f"Total trades           : {total}")
    print(f"Short trades (<{HOLDING_PERIOD} days) : {short}")

    if short == 0:
        print("\nPERFECT: All trades held for full duration.")
    else:
        print("\nWARNING: Some trades were exited early.")
        print(trade_df[trade_df["duration_days"] < HOLDING_PERIOD].head())

    # Save summary
    with open(f"{RESULT_DIR}/summary.txt", "w") as f:
        f.write(f"Total Trades: {total}\nShort Trades: {short}\n")
        if short == 0:
            f.write("PERFECT RUN.\n")


if __name__ == "__main__":
    run_backtest()
