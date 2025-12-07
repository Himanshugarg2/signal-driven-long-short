# Signal-Driven Long-Short: Data Pipeline

This project generates and processes synthetic financial datasets for signal-driven long-short strategies.

## Python Version

- **Recommended Python version:** 3.9 

## 1. Setup: Create and Activate Virtual Environment

**Windows:**
```powershell
C:\Python39\python.exe -m venv venv
.\venv\Scripts\activate
```


## 2. Install Requirements
```bash
pip install -r requirements.txt
```

## 3. Generate Synthetic Data
From the `Scripts` directory:
```bash
python generate_data.py
```
- Output: `data/synthetic_raw.csv`

## 4. Clean the Dataset
```bash
python clean_dataset.py
```
- Output: `data/synthetic_clean.csv`

## 5. Flag the Dataset
```bash
python flag_dataset.py
```
- Output: `data/synthetic_flagged.csv`

## 6. Generate Trading Dataset for Vector/Backtest
```bash
python build_trading_dataset.py
```
- Outputs (in `data/trading/`):
  - `trading_prices.csv`
  - `trading_signals.csv`
  - `trading_weights.csv`

## 7. Run Backtest
```bash
python run_backtest.py
```
- Outputs (in `results/`):
  - `backtest_stats.csv`
  - `equity_curve.csv`
  - `summary.txt`
  - `trade_log.csv`

---

**Tip:**
- Run each script from the `Scripts` directory for correct relative paths.
- Review and edit scripts as needed for custom logic or parameters.
