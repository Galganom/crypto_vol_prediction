"""Z-score calibration test for BTC 4h forecasts.

For each test bar with predicted RV_4h, treat sqrt(pred_RV) as the forecast
standard deviation of the 4h log return.  The standardized residual

    z_t = r_{t→t+4h} / sqrt( pred_RV_t )

should be approximately ~N(0,1) under a calibrated variance forecast and
normal-ish returns.  We compare TS-only vs Main Model against this benchmark.

Reports per model:
  mean, std, skew, kurtosis
  fraction beyond 1σ, 1.96σ, 2.58σ, 3σ, 3.29σ
  KS-test against N(0,1)
  Jarque-Bera normality test
  Kupiec POF test on the 5% two-tailed coverage

Pools all 6 windows = 104,646 4h bars.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kstest, jarque_bera, chi2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from main_model_common import build_price_bars, align_bars_with_news, load_news_features

# Defaults; overridable via --pred-dir / --out-dir (see main()).
PRED_DIR = REPO_ROOT / "results/main/btc_cryptobert_v2/predictions"
OUT_DIR  = REPO_ROOT / "results/analysis/btc_news_impact_v3"
HORIZON_BARS = 16   # 15-min bars per 4h


def forward_4h_log_return(bars):
    """Sum of next 16 15-min log returns, aligned to each bar's index."""
    log_ret = bars["log_ret"].values.astype(np.float64)
    return (pd.Series(log_ret).shift(-1).rolling(HORIZON_BARS, min_periods=HORIZON_BARS)
              .sum().shift(-(HORIZON_BARS - 1))).values


def load_pooled():
    bars = build_price_bars("BTC")
    news = load_news_features("BTC")
    bars, _, _ = align_bars_with_news(bars, news["timestamps"])
    fwd_full = forward_4h_log_return(bars)

    ts, mn, ac, idx, dt = [], [], [], [], []
    for f in sorted(PRED_DIR.glob("window_*_4h_predictions.npz")):
        d = np.load(f, allow_pickle=True)
        ts.append(d["ts_mlp"]); mn.append(d["main_model"]); ac.append(d["actual"])
        idx.append(d["test_idx"]); dt.append(d["datetime"])
    ts = np.concatenate(ts).astype(np.float64)
    mn = np.concatenate(mn).astype(np.float64)
    ac = np.concatenate(ac).astype(np.float64)
    idx = np.concatenate(idx).astype(np.int64)
    fwd = fwd_full[idx]
    return ts, mn, ac, fwd


def coverage(z, k):
    valid = np.isfinite(z)
    return float((np.abs(z[valid]) > k).mean())


def kupiec_pof(z, alpha=0.05):
    """Kupiec proportion-of-failures test for two-tailed coverage at level alpha."""
    valid = np.isfinite(z)
    n = int(valid.sum())
    crit = 1.959963984540054  # 95% z-bound
    x = int((np.abs(z[valid]) > crit).sum())
    expected = alpha * n
    if x == 0 or x == n:
        return float("nan"), float("nan"), x, n, expected
    pi_hat = x / n
    LR = 2 * (x * np.log(pi_hat / alpha) + (n - x) * np.log((1 - pi_hat) / (1 - alpha)))
    p = 1.0 - chi2.cdf(LR, df=1)
    return float(LR), float(p), x, n, expected


def summary(z, name):
    z = z[np.isfinite(z)]
    n = len(z)
    mean, std = z.mean(), z.std(ddof=1)
    skew = float(((z - mean) ** 3).mean() / (std ** 3))
    kurt = float(((z - mean) ** 4).mean() / (std ** 4)) - 3.0
    ks_stat, ks_p = kstest(z, "norm")
    jb_stat, jb_p = jarque_bera(z)
    LR, kp, breaches, _, exp_b = kupiec_pof(z, alpha=0.05)
    cov = {
        ">1σ":     coverage(z, 1.0),
        ">1.96σ":  coverage(z, 1.959963984540054),
        ">2.58σ":  coverage(z, 2.575829303548901),
        ">3σ":     coverage(z, 3.0),
        ">3.29σ":  coverage(z, 3.290526731492022),
    }
    return {
        "model": name, "N": n,
        "mean": mean, "std": std, "skew": skew, "excess_kurt": kurt,
        "ks_stat": ks_stat, "ks_p": ks_p,
        "jb_stat": jb_stat, "jb_p": jb_p,
        "kupiec_LR": LR, "kupiec_p": kp,
        "breaches": breaches, "expected_breaches": exp_b,
        **cov,
    }


def main():
    import argparse
    global PRED_DIR, OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", default=str(PRED_DIR),
                        help="Directory of window_*_4h_predictions.npz.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    PRED_DIR = Path(args.pred_dir)
    OUT_DIR  = Path(args.out_dir)
    print(f"Pred dir: {PRED_DIR}")
    print(f"Out dir:  {OUT_DIR}")
    ts, mn, ac, fwd = load_pooled()
    print(f"Pooled test bars: {len(ts):,}")
    print(f"Forward returns NaN: {int((~np.isfinite(fwd)).sum()):,}")

    sigma_ts   = np.sqrt(np.clip(ts, 1e-15, None))
    sigma_mn   = np.sqrt(np.clip(mn, 1e-15, None))
    sigma_act  = np.sqrt(np.clip(ac, 1e-15, None))

    z_ts   = fwd / sigma_ts
    z_main = fwd / sigma_mn
    z_orac = fwd / sigma_act   # if we knew realized vol perfectly

    rows = [summary(z_ts, "TS-only"),
            summary(z_main, "Main Model"),
            summary(z_orac, "Oracle (realized vol)")]

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    benchmarks = {
        ">1σ":    0.3173,
        ">1.96σ": 0.0500,
        ">2.58σ": 0.0100,
        ">3σ":    0.0027,
        ">3.29σ": 0.0010,
    }
    print("\n=== Tail-coverage breach summary (smaller |actual - expected| = better calibrated) ===")
    print(f"{'Threshold':>10s} {'Expected':>10s}  {'TS-only':>9s}  {'Main':>9s}  {'Oracle':>9s}")
    for k, exp_p in benchmarks.items():
        ts_p = next(r for r in rows if r['model']=='TS-only')[k]
        mn_p = next(r for r in rows if r['model']=='Main Model')[k]
        or_p = next(r for r in rows if r['model']=='Oracle (realized vol)')[k]
        print(f"{k:>10s} {exp_p*100:>9.2f}%  {ts_p*100:>8.2f}%  "
              f"{mn_p*100:>8.2f}%  {or_p*100:>8.2f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "zscore_results_4h.csv", index=False, float_format="%.6f")
    print(f"\nSaved: {OUT_DIR/'zscore_results_4h.csv'}")


if __name__ == "__main__":
    main()
