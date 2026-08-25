"""Z-score calibration test across all (asset, horizon) combinations.

For each test bar:  z_t = r_{t→t+H} / sqrt( pred_RV_t )

Under perfect calibration with Gaussian returns:  z ~ N(0, 1).
We bucket z into 0.5σ-wide bins from -3σ to +3σ and compare observed
fractions to Gaussian expectations.

Configurations:  BTC + ETH × 1h + 4h horizons.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, kstest, jarque_bera

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from main_model_common import build_price_bars, align_bars_with_news, load_news_features

OUT_DIR = REPO_ROOT / "results/analysis/btc_news_impact_v3"
BAR_MIN = 15
HORIZON_BARS = {"1h": 4, "4h": 16, "8h": 32}
ASSETS = {
    "BTC": REPO_ROOT / "results/main/btc_cryptobert_v2/predictions",
    "ETH": REPO_ROOT / "results/main/eth_cryptobert_v2/predictions",
}
HORIZONS = ["1h", "4h"]

# Bucket boundaries every 0.5σ
EDGES = np.arange(-3.0, 3.01, 0.5)


def forward_log_return(bars, n_bars):
    log_ret = bars["log_ret"].values.astype(np.float64)
    return (pd.Series(log_ret).shift(-1)
              .rolling(n_bars, min_periods=n_bars).sum()
              .shift(-(n_bars - 1))).values


def load_pool(asset, horizon):
    bars = build_price_bars(asset)
    news = load_news_features(asset)
    bars, _, _ = align_bars_with_news(bars, news["timestamps"])
    fwd_full = forward_log_return(bars, HORIZON_BARS[horizon])

    files = sorted(ASSETS[asset].glob(f"window_*_{horizon}_predictions.npz"))
    ts_list, mn_list, ac_list, idx_list = [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        ts_list.append(d["ts_mlp"]); mn_list.append(d["main_model"])
        ac_list.append(d["actual"]); idx_list.append(d["test_idx"])
    ts = np.concatenate(ts_list).astype(np.float64)
    mn = np.concatenate(mn_list).astype(np.float64)
    ac = np.concatenate(ac_list).astype(np.float64)
    idx = np.concatenate(idx_list).astype(np.int64)
    fwd = fwd_full[idx]
    return ts, mn, ac, fwd, len(files)


def bucketize(z, edges):
    """Return counts and % in each [edges[i], edges[i+1]) plus left/right tail."""
    z = z[np.isfinite(z)]
    n = len(z)
    out = []
    cnt = int((z < edges[0]).sum())
    out.append({"bin": f"< {edges[0]:.1f}", "lo": -np.inf, "hi": edges[0],
                "obs_count": cnt, "obs_pct": cnt/max(n,1)*100})
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i+1]
        cnt = int(((z >= lo) & (z < hi)).sum())
        out.append({"bin": f"[{lo:.1f}, {hi:.1f})", "lo": lo, "hi": hi,
                    "obs_count": cnt, "obs_pct": cnt/max(n,1)*100})
    cnt = int((z >= edges[-1]).sum())
    out.append({"bin": f"≥ {edges[-1]:.1f}", "lo": edges[-1], "hi": np.inf,
                "obs_count": cnt, "obs_pct": cnt/max(n,1)*100})
    for r in out:
        lo, hi = r["lo"], r["hi"]
        exp_p = norm.cdf(hi) - norm.cdf(lo)
        r["exp_count"] = exp_p * n
        r["exp_pct"]   = exp_p * 100
        r["diff_pct"]  = r["obs_pct"] - r["exp_pct"]
    return pd.DataFrame(out), n


def stats_block(z):
    z = z[np.isfinite(z)]
    n = len(z)
    if n < 2: return None
    mean, std = float(z.mean()), float(z.std(ddof=1))
    skew = float(((z - mean) ** 3).mean() / (std ** 3))
    kurt = float(((z - mean) ** 4).mean() / (std ** 4)) - 3.0
    ks_stat, ks_p = kstest(z, "norm")
    jb_stat, jb_p = jarque_bera(z)
    return {"N": n, "mean": mean, "std": std, "skew": skew,
            "excess_kurt": kurt, "ks_stat": float(ks_stat),
            "jb_stat": float(jb_stat)}


def run_one(asset, horizon):
    ts, mn, ac, fwd, n_win = load_pool(asset, horizon)
    sigma_ts = np.sqrt(np.clip(ts, 1e-15, None))
    sigma_mn = np.sqrt(np.clip(mn, 1e-15, None))
    sigma_ac = np.sqrt(np.clip(ac, 1e-15, None))
    z_ts = fwd / sigma_ts
    z_mn = fwd / sigma_mn
    z_or = fwd / sigma_ac

    print(f"\n{'='*88}")
    print(f"  {asset}  {horizon}  —  pooled {len(ts):,} bars across {n_win} windows")
    print(f"{'='*88}")
    print("\nSummary statistics (target: mean=0, std=1, skew=0, excess_kurt=0):")
    s_ts = stats_block(z_ts); s_mn = stats_block(z_mn); s_or = stats_block(z_or)
    summary_rows = []
    for name, s in [("TS-only", s_ts), ("Main Model", s_mn), ("Oracle", s_or)]:
        s2 = {"model": name, **s}
        summary_rows.append(s2)
        print(f"  {name:11s}: N={s['N']:7,}  mean={s['mean']:+.4f}  "
              f"std={s['std']:.4f}  skew={s['skew']:+.3f}  "
              f"excess_kurt={s['excess_kurt']:+.3f}  KS={s['ks_stat']:.4f}")

    df_ts, _ = bucketize(z_ts, EDGES)
    df_mn, _ = bucketize(z_mn, EDGES)
    df_or, _ = bucketize(z_or, EDGES)
    out = df_ts[["bin"]].copy()
    out["Expected N"]   = df_ts["exp_count"].round(0).astype(int).values
    out["TS N"]         = df_ts["obs_count"].astype(int).values
    out["Main N"]       = df_mn["obs_count"].astype(int).values
    out["Oracle N"]     = df_or["obs_count"].astype(int).values
    out["Expected %"]   = df_ts["exp_pct"].round(3).values
    out["TS %"]         = df_ts["obs_pct"].round(3).values
    out["Main %"]       = df_mn["obs_pct"].round(3).values
    out["Oracle %"]     = df_or["obs_pct"].round(3).values

    print(f"\nZ-score bucket counts and fractions:")
    print(out.to_string(index=False))

    out["asset"] = asset; out["horizon"] = horizon
    summary_df = pd.DataFrame(summary_rows)
    summary_df["asset"] = asset; summary_df["horizon"] = horizon
    return out, summary_df


def main():
    bucket_dfs, summary_dfs = [], []
    for asset in ASSETS:
        for h in HORIZONS:
            b, s = run_one(asset, h)
            bucket_dfs.append(b); summary_dfs.append(s)
    bucket_all = pd.concat(bucket_dfs, ignore_index=True)
    summary_all = pd.concat(summary_dfs, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bucket_all.to_csv(OUT_DIR / "zscore_buckets_all.csv", index=False, float_format="%.4f")
    summary_all.to_csv(OUT_DIR / "zscore_summary_all.csv", index=False, float_format="%.6f")
    print(f"\nSaved: {OUT_DIR/'zscore_buckets_all.csv'}")
    print(f"Saved: {OUT_DIR/'zscore_summary_all.csv'}")


if __name__ == "__main__":
    main()
