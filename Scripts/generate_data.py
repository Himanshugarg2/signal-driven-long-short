"""
generate_data.py

Generates synthetic daily data for a universe of companies with vanish events.

Output:
- data/synthetic_raw.csv

Key behavior:
- 10 years of business days
- Always 50 tickers active each trading day
- Vanish events occur every random 2–5 days
- Each vanish event removes 5–9 tickers:
    - at least 1 top 10%
    - at least 1 bottom 10%
    - at least 1 middle group
- Vanished tickers trade on the vanish day and disappear the next day
- Replacement tickers appear the next day
- Signals follow AR(1), prices follow random walk
"""

import os
import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Configurable parameters
# -----------------------------
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
INITIAL_UNIVERSE = 50

VANISH_GAP_OPTIONS = [2, 3, 4, 5]
VANISH_BATCH_MIN = 5
VANISH_BATCH_MAX = 9
TOP_PERCENTILE = 0.10
BOTTOM_PERCENTILE = 0.10

SEED = 42
OUTPUT_DIR = "data"

np.random.seed(SEED)
random.seed(SEED)


# -----------------------------
# Utility helpers
# -----------------------------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def business_days(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, end=end)


def next_ticker_name(next_id: int) -> str:
    return f"C{next_id}"


# -----------------------------
# Ticker state + simulation
# -----------------------------
class TickerState:
    def __init__(self, ticker: str, init_price: float, init_signal: float):
        self.ticker = ticker
        self.last_price = init_price
        self.last_signal = init_signal

        self.phi = np.clip(np.random.normal(0.6, 0.1), 0.2, 0.95)
        self.drift = np.random.normal(0.0002, 0.0005)
        self.vol = np.clip(np.random.normal(0.02, 0.01), 0.005, 0.08)


def simulate_next(state: TickerState):
    """Simulate next day's close price and signal."""
    eps = np.random.normal(0, 0.5)

    next_signal = (
        state.phi * state.last_signal
        + (1 - state.phi) * np.random.normal(0, 0.1)
        + eps * 0.05
    )

    shock = np.random.normal(0, state.vol)
    next_price = state.last_price * (1 + state.drift + shock)

    if next_price <= 0:
        next_price = max(0.5, state.last_price * 0.5)

    state.last_price = next_price
    state.last_signal = next_signal

    return float(next_price), float(next_signal)


# -----------------------------
# Universe initialization
# -----------------------------
def initialize_universe(n: int, start_id: int = 1):
    states = {}
    next_id = start_id

    for _ in range(n):
        name = next_ticker_name(next_id)
        price = float(np.random.uniform(50, 200))
        signal = float(np.random.normal(0, 1))
        states[name] = TickerState(name, price, signal)
        next_id += 1

    return states, next_id


# -----------------------------
# Vanish selection (today's signal only)
# -----------------------------
def select_vanish_batch(active_tickers, history_signals, batch_size):

    today_signal = {t: history_signals[t][-1][1] for t in active_tickers}

    ranked = sorted(active_tickers, key=lambda t: today_signal[t], reverse=True)

    n = len(ranked)
    top_k = max(1, math.ceil(n * TOP_PERCENTILE))
    bottom_k = max(1, math.ceil(n * BOTTOM_PERCENTILE))

    top_group = set(ranked[:top_k])
    bottom_group = set(ranked[-bottom_k:])
    middle_group = set(ranked[top_k : n - bottom_k])

    chosen = []
    used = set()

    # must include 1 top, 1 bottom, 1 mid
    t = random.choice(list(top_group))
    chosen.append((t, "top"))
    used.add(t)

    t = random.choice(list(bottom_group - used))
    chosen.append((t, "bottom"))
    used.add(t)

    if len(middle_group - used) > 0:
        t = random.choice(list(middle_group - used))
        chosen.append((t, "mid"))
        used.add(t)

    # fill remaining slots
    remaining = batch_size - len(chosen)
    candidates = list(set(active_tickers) - used)

    if remaining > 0:
        sample = random.sample(candidates, remaining)
        for t in sample:
            group = (
                "top" if t in top_group else "bottom" if t in bottom_group else "mid"
            )
            chosen.append((t, group))

    return chosen


# -----------------------------
# Main generator (only synthetic_raw.csv saved)
# -----------------------------
def generate_synthetic_dataset(
    start_date=START_DATE,
    end_date=END_DATE,
    initial_universe=INITIAL_UNIVERSE,
    vanish_gap_options=VANISH_GAP_OPTIONS,
    vanish_batch_min=VANISH_BATCH_MIN,
    vanish_batch_max=VANISH_BATCH_MAX,
):
    ensure_dir(OUTPUT_DIR)

    dates = business_days(start_date, end_date)

    states, next_id = initialize_universe(initial_universe)
    active = list(states.keys())

    history_records = []
    history_signals = defaultdict(list)

    next_vanish_day = random.choice(vanish_gap_options)
    to_remove_next_day = set()

    for i, today in enumerate(dates):

        # Remove vanished tickers
        for t in list(to_remove_next_day):
            active.remove(t)
            del states[t]
        to_remove_next_day.clear()

        # Generate today's tick data
        for t in list(active):
            price, signal = simulate_next(states[t])
            history_records.append((today, t, price, signal))
            history_signals[t].append((today, signal))

        # Vanish event today?
        if i == next_vanish_day:

            batch_size = random.randint(vanish_batch_min, vanish_batch_max)
            batch_size = min(batch_size, len(active) - 1)

            chosen = select_vanish_batch(active, history_signals, batch_size)

            # remove tomorrow
            to_remove_next_day = {t for t, _ in chosen}

            # add replacements
            for _ in range(len(chosen)):
                name = next_ticker_name(next_id)
                next_id += 1
                states[name] = TickerState(
                    name,
                    float(np.random.uniform(50, 200)),
                    float(np.random.normal(0, 1)),
                )
                active.append(name)

            # schedule next vanish event
            next_vanish_day = i + random.choice(vanish_gap_options)
            if next_vanish_day >= len(dates):
                next_vanish_day = -1

    # Final dataframe (UPDATED SORTING BY SIGNAL)
    df = pd.DataFrame(history_records, columns=["date", "ticker", "close", "signal"])

    # Sort by date, then signal DESCENDING (highest signal first)
    df.sort_values(["date", "signal"], ascending=[True, False], inplace=True)

    df.to_csv(os.path.join(OUTPUT_DIR, "synthetic_raw.csv"), index=False)
    print("Saved synthetic_raw.csv")

    return df


if __name__ == "__main__":
    df = generate_synthetic_dataset()
