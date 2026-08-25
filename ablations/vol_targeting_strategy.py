"""Volatility-targeting backtest across the 5 production models, BTC + ETH, 4h.

Strategy: at each test bar t, hold a position
    w_t = min(target_ann_vol / pred_ann_vol_t, MAX_LEV)
Realize the next-4h log return; portfolio return = w_t · r_{t→t+4h}.
Rebalance every 4h (one-step look-ahead, no compounding within a bar).

Compares HAR / EGARCH / EGARCH+NED / TS MLP / Main Model plus:
  - buy_and_hold  : w_t = 1 always
  - oracle        : uses the actual realized RV (upper bound)

Outputs per asset:
  results/strategy/vol_target_{btc,eth}_4h/
    per_model_metrics.csv     Sharpe, ann vol, ann ret, max DD, turnover, hit rate
    equity_curves.csv         cumulative portfolio return per model
    equity_curves.png         visual comparison
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from main_model_common import build_price_bars, align_bars_with_news, load_news_features  # noqa: E402

# ─── Config ──────────────────────────────────────────────────────────────────
HORIZON_HOURS = 4
PERIODS_PER_YEAR = 365 * 24 / HORIZON_HOURS   # 2190 for 4h
TARGET_ANN_VOL = 0.20                          # 20% annualized
MAX_LEV = 3.0                                  # cap leverage
TRANSACTION_COST_BPS = 0                       # 0 bps per side per rebalance
MODELS = ["har", "egarch", "egarch_ned", "ts_mlp", "main_model"]
DISPLAY = {
    "har": "HAR",
    "egarch": "EGARCH",
    "egarch_ned": "EGARCH+NED",
    "ts_mlp": "TS MLP",
    "main_model": "Main Model",
    "oracle": "Oracle (actual RV)",
    "buy_and_hold": "Buy & Hold",
}
COLORS = {
    "har":          "#9CA3AF",
    "egarch":       "#6B7280",
    "egarch_ned":   "#F59E0B",
    "ts_mlp":       "#3B82F6",
    "main_model":   "#DC2626",
    "oracle":       "#10B981",
    "buy_and_hold": "#000000",
}
DEFAULT_ASSETS = {
    "BTC": REPO_ROOT / "results/main/btc_cryptobert_v2",
    "ETH": REPO_ROOT / "results/main/eth_cryptobert_v2",
}
OUT_ROOT = REPO_ROOT / "results/strategy"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def build_forward_log_returns(bars):
    """Return forward 4h log returns aligned to each bar."""
    log_ret = bars["log_ret"].values.astype(np.float64)
    fwd = pd.Series(log_ret).shift(-1).rolling(16, min_periods=16).sum().shift(-15).values
    # equivalent to sum of log_ret over t+1 .. t+16
    return fwd


def vol_target_position(pred_rv, target_ann_vol=TARGET_ANN_VOL, max_lev=MAX_LEV):
    """pred_rv → annualized vol → vol-targeted weight (capped)."""
    pred_rv = np.clip(pred_rv, 1e-12, None)
    pred_vol_ann = np.sqrt(pred_rv * PERIODS_PER_YEAR)
    w = target_ann_vol / pred_vol_ann
    return np.clip(w, 0.0, max_lev)


def compute_metrics(weights, fwd_returns, target_ann_vol=TARGET_ANN_VOL,
                    tc_bps=TRANSACTION_COST_BPS):
    """Sharpe, ann vol, ann ret, max DD, turnover, target-hit ratio."""
    w = np.asarray(weights, dtype=np.float64)
    r = np.asarray(fwd_returns, dtype=np.float64)
    valid = np.isfinite(w) & np.isfinite(r)
    w = w[valid]; r = r[valid]
    # Turnover: |w_t - w_{t-1}|, transaction cost in log-return units
    dw = np.abs(np.diff(w, prepend=w[0]))
    tc = dw * (tc_bps / 1e4)
    port_ret = w * r - tc
    mean_ret = port_ret.mean()
    std_ret = port_ret.std(ddof=1)
    ann_ret = mean_ret * PERIODS_PER_YEAR
    ann_vol = std_ret * np.sqrt(PERIODS_PER_YEAR)
    sharpe = ann_ret / max(ann_vol, 1e-12)
    cum = np.cumsum(port_ret)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak)              # in log space; max drawdown = min(dd)
    max_dd = abs(dd.min())
    target_hit_ratio = (ann_vol / target_ann_vol) if target_ann_vol > 0 else 0.0
    turnover_avg = dw.mean()
    return {
        "ann_ret": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "vol_ratio": float(target_hit_ratio),   # realized / target — should be ~1
        "turnover_avg": float(turnover_avg),
        "n_obs": int(len(port_ret)),
    }


def load_predictions(asset_dir):
    """Returns dict: model_name -> pooled prediction array, plus 'actual',
    'datetime', and pooled aligned forward returns."""
    files = sorted(asset_dir.glob("predictions/window_*_4h_predictions.npz"))
    out = {m: [] for m in MODELS}
    out["actual"] = []
    out["test_idx"] = []
    out["datetime"] = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        for m in MODELS:
            out[m].append(d[m])
        out["actual"].append(d["actual"])
        out["test_idx"].append(d["test_idx"])
        out["datetime"].append(d["datetime"])
    for k in list(out.keys()):
        out[k] = np.concatenate(out[k])
    return out, files


def run_asset(asset, asset_dir, out_suffix=""):
    asset_dir = Path(asset_dir)
    if not asset_dir.exists():
        print(f"[{asset}] No results at {asset_dir}; skipping.")
        return None
    label = f"vol_target_{asset.lower()}_4h" + (f"_{out_suffix}" if out_suffix else "")
    out_dir = OUT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)

    preds, files = load_predictions(asset_dir)
    n = len(preds["actual"])
    print(f"\n[{asset}] pooled test rows: {n:,} across {len(files)} windows")

    # Build forward 4h log returns
    bars = build_price_bars(asset)
    news = load_news_features(asset)
    bars, _, _ = align_bars_with_news(bars, news["timestamps"])
    fwd = build_forward_log_returns(bars)
    fwd_aligned = fwd[preds["test_idx"]]

    # Strategy per model
    rows = []
    equity = {}
    for m in MODELS:
        w = vol_target_position(preds[m])
        port_ret = w * fwd_aligned
        # transaction costs subtracted in compute_metrics
        metrics = compute_metrics(w, fwd_aligned)
        rows.append({"model": DISPLAY[m], **metrics})
        equity[DISPLAY[m]] = np.cumsum(w * fwd_aligned - np.abs(np.diff(w, prepend=w[0])) * (TRANSACTION_COST_BPS / 1e4))

    # Oracle (uses realized RV — upper bound on what timing of vol-target can do)
    w_oracle = vol_target_position(preds["actual"])
    oracle_metrics = compute_metrics(w_oracle, fwd_aligned)
    rows.append({"model": DISPLAY["oracle"], **oracle_metrics})
    equity[DISPLAY["oracle"]] = np.cumsum(
        w_oracle * fwd_aligned - np.abs(np.diff(w_oracle, prepend=w_oracle[0])) * (TRANSACTION_COST_BPS / 1e4)
    )

    # Buy and hold (always w=1, no rebalance after open)
    w_bh = np.ones_like(fwd_aligned)
    bh_metrics = compute_metrics(w_bh, fwd_aligned)
    rows.append({"model": DISPLAY["buy_and_hold"], **bh_metrics})
    equity[DISPLAY["buy_and_hold"]] = np.cumsum(fwd_aligned)

    df = pd.DataFrame(rows)
    df["sharpe"] = df["sharpe"].round(3)
    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    df.to_csv(out_dir / "per_model_metrics.csv", index=False, float_format="%.6f")
    print(df.to_string(index=False))

    # Save equity curves
    dt = pd.to_datetime(preds["datetime"], utc=True)
    eq_df = pd.DataFrame({"datetime": dt})
    for name, eq in equity.items():
        eq_df[name] = eq
    eq_df.to_csv(out_dir / "equity_curves.csv", index=False, float_format="%.6f")

    # Plot
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titleweight": "bold"})
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_order = ["Buy & Hold", "HAR", "EGARCH", "EGARCH+NED", "TS MLP", "Main Model", "Oracle (actual RV)"]
    for label in plot_order:
        if label in equity:
            key = next(k for k, v in DISPLAY.items() if v == label)
            ax.plot(dt, equity[label], label=label, color=COLORS[key],
                    linewidth=2 if label in ("Main Model", "Oracle (actual RV)") else 1.4,
                    alpha=0.95)
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_title(f"{asset} 4h — Vol-targeted equity curves (target {int(TARGET_ANN_VOL*100)}% ann. vol, lev cap {MAX_LEV}x, {TRANSACTION_COST_BPS}bps tc)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative log-return")
    ax.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "equity_curves.png", dpi=200)
    plt.close()
    print(f"Saved: {out_dir}/")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs="+", default=["BTC", "ETH"])
    parser.add_argument("--btc-dir", default=str(DEFAULT_ASSETS["BTC"]),
                        help="Predictions dir for BTC (default: btc_cryptobert_v2)")
    parser.add_argument("--eth-dir", default=str(DEFAULT_ASSETS["ETH"]),
                        help="Predictions dir for ETH (default: eth_cryptobert_v2)")
    parser.add_argument("--out-suffix", default="",
                        help="Suffix appended to output folder name "
                             "(e.g. 'v3' -> vol_target_btc_4h_v3).")
    args = parser.parse_args()
    asset_dirs = {"BTC": Path(args.btc_dir), "ETH": Path(args.eth_dir)}
    results = {}
    for a in args.assets:
        df = run_asset(a, asset_dirs[a], out_suffix=args.out_suffix)
        if df is not None:
            results[a] = df
    # Cross-asset summary
    if len(results) > 1:
        combo = pd.concat(
            [df.assign(asset=a) for a, df in results.items()], ignore_index=True
        )
        combo = combo[["asset", "model", "sharpe", "ann_ret", "ann_vol",
                        "max_dd", "vol_ratio", "turnover_avg", "n_obs"]]
        out_csv = OUT_ROOT / "summary_btc_eth.csv"
        combo.to_csv(out_csv, index=False, float_format="%.6f")
        print(f"\nCross-asset summary: {out_csv}")
        print(combo.to_string(index=False))


if __name__ == "__main__":
    main()
