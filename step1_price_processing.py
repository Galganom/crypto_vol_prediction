"""
Step 1: Price Data Processing
- Load 1-min BTC/USD candles (full 2012-2025 history)
- Aggregate to 5-min bars
- Calculate forward realized volatility targets (RV_1h, RV_4h)
- Calculate lagged realized volatility features (rv_12, rv_48, rv_144, rv_288)
"""

import pandas as pd
import numpy as np

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/price/btcusd_bitstamp_1min_2012-2025.csv.gz")
df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
df = df.sort_values("datetime").reset_index(drop=True)

print(f"Raw 1-min data: {len(df):,} rows")
print(f"Date range: {df['datetime'].min()} → {df['datetime'].max()}")

# ── Aggregate to 5-min bars ───────────────────────────────────────────────────
df = df.set_index("datetime")
bars = df.resample("5min").agg({
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "timestamp": "first",
}).dropna(subset=["close"])

bars = bars.reset_index()
print(f"\n5-min bars: {len(bars):,} rows")

# ── Log returns ───────────────────────────────────────────────────────────────
bars["log_ret"] = np.log(bars["close"] / bars["close"].shift(1))
bars["log_ret_sq"] = bars["log_ret"] ** 2

# ── Forward RV targets (vectorized) ──────────────────────────────────────────
# RV at bar i = sum of log_ret_sq from i+1 to i+W (inclusive)
# Use reversed cumsum trick: reverse, rolling sum, reverse back, then shift
sq = bars["log_ret_sq"].values
# Forward rolling sum: shift by -1 first (exclude current bar), then rolling
sq_shifted = pd.Series(sq).shift(-1)
bars["RV_1h"] = sq_shifted.iloc[::-1].rolling(window=12, min_periods=12).sum().iloc[::-1].values
bars["RV_4h"] = sq_shifted.iloc[::-1].rolling(window=48, min_periods=48).sum().iloc[::-1].values

# ── Lagged RV features ──────────────────────────────────────────────────────
bars["rv_12"] = bars["log_ret_sq"].rolling(window=12, min_periods=12).sum()
bars["rv_48"] = bars["log_ret_sq"].rolling(window=48, min_periods=48).sum()
bars["rv_144"] = bars["log_ret_sq"].rolling(window=144, min_periods=144).sum()
bars["rv_288"] = bars["log_ret_sq"].rolling(window=288, min_periods=288).sum()

# ── Drop NaN rows ─────────────────────────────────────────────────────────────
cols_needed = ["RV_1h", "RV_4h", "rv_12", "rv_48", "rv_144", "rv_288"]
bars_clean = bars.dropna(subset=cols_needed).reset_index(drop=True)

print(f"After dropping NaN: {len(bars_clean):,} rows")
print(f"Date range: {bars_clean['datetime'].min()} → {bars_clean['datetime'].max()}")

# ── Stats ─────────────────────────────────────────────────────────────────────
print("\n── Target distributions ──")
for col in ["RV_1h", "RV_4h"]:
    v = bars_clean[col]
    print(f"\n{col}:")
    print(f"  mean:   {v.mean():.6e}")
    print(f"  median: {v.median():.6e}")
    print(f"  std:    {v.std():.6e}")
    print(f"  min:    {v.min():.6e}")
    print(f"  max:    {v.max():.6e}")
    print(f"  p90:    {v.quantile(0.9):.6e}")
    print(f"  p99:    {v.quantile(0.99):.6e}")

print("\n── Lagged feature distributions ──")
for col in ["rv_12", "rv_48", "rv_144", "rv_288"]:
    v = bars_clean[col]
    print(f"{col:>8s}: mean={v.mean():.6e}  median={v.median():.6e}  max={v.max():.6e}")

# ── Save ──────────────────────────────────────────────────────────────────────
bars_clean.to_parquet("data/btc_5min_bars.parquet", index=False)
print(f"\nSaved to data/btc_5min_bars.parquet ({len(bars_clean):,} rows)")
