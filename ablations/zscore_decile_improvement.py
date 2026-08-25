"""Main Model vs TS-only forecast accuracy, sliced by z-score decile.

For each pooled OOS 4h bar we form the TS-only standardized return
    z_t = r_{t->t+4h} / sqrt(pred_RV_TS_t)
and split all bars into 10 equal-frequency deciles of (signed) z.

Within each decile we compute R2 / QLIKE / MSE (repo conventions:
log-RV MSE, ratio QLIKE, log-RV R2) for TS-only and for the Main Model,
then the Main-vs-TS improvement:
    dR2          = R2_main - R2_ts                 (+ = Main better)
    QLIKE_improv = 100*(QLIKE_ts - QLIKE_main)/QLIKE_ts
    MSE_improv   = 100*(MSE_ts   - MSE_main)/MSE_ts

This shows in which return-shock regimes the news head helps.

Outputs (default --out-dir results/analysis/btc_cryptobert_v3):
    zscore_decile_improvement.csv
    zscore_decile_improvement.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from main_model_common import (  # noqa: E402
    align_bars_with_news, build_price_bars, compute_metrics, load_news_features,
)

DEFAULT_PRED_DIR = REPO_ROOT / "results/main/btc_cryptobert_v3/predictions"
DEFAULT_OUT_DIR = REPO_ROOT / "results/analysis/btc_cryptobert_v3"
HORIZON_BARS = 16   # 15-min bars per 4h
N_DECILES = 10


def forward_4h_log_return(bars):
    log_ret = bars["log_ret"].values.astype(np.float64)
    return (pd.Series(log_ret).shift(-1)
              .rolling(HORIZON_BARS, min_periods=HORIZON_BARS)
              .sum().shift(-(HORIZON_BARS - 1))).values


def load_pooled(pred_dir):
    bars = build_price_bars("BTC")
    news = load_news_features("BTC")
    bars, _, _ = align_bars_with_news(bars, news["timestamps"])
    fwd_full = forward_4h_log_return(bars)

    ts, mn, ac, idx = [], [], [], []
    for f in sorted(pred_dir.glob("window_*_4h_predictions.npz")):
        d = np.load(f, allow_pickle=True)
        ts.append(d["ts_mlp"]); mn.append(d["main_model"])
        ac.append(d["actual"]); idx.append(d["test_idx"])
    ts = np.concatenate(ts).astype(np.float64)
    mn = np.concatenate(mn).astype(np.float64)
    ac = np.concatenate(ac).astype(np.float64)
    idx = np.concatenate(idx).astype(np.int64)
    fwd = fwd_full[idx]
    return ts, mn, ac, fwd


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", default=str(DEFAULT_PRED_DIR),
                        help="Directory of window_*_4h_predictions.npz.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pred dir: {pred_dir}")
    print(f"Out dir:  {out_dir}")

    ts, mn, ac, fwd = load_pooled(pred_dir)
    z_ts = fwd / np.sqrt(np.clip(ts, 1e-15, None))
    ok = np.isfinite(z_ts) & np.isfinite(ac) & np.isfinite(ts) & np.isfinite(mn)
    ts, mn, ac, z = ts[ok], mn[ok], ac[ok], z_ts[ok]
    print(f"Pooled bars: {len(z):,}  (dropped {int((~ok).sum()):,} non-finite)")

    # Equal-frequency signed-z deciles
    dec = pd.qcut(z, N_DECILES, labels=False, duplicates="drop")

    rows = []
    for d in range(N_DECILES):
        m = dec == d
        n = int(m.sum())
        if n == 0:
            continue
        mt = compute_metrics(ts[m], ac[m])
        mm = compute_metrics(mn[m], ac[m])
        rows.append({
            "decile": d + 1,
            "N": n,
            "z_lo": float(z[m].min()),
            "z_hi": float(z[m].max()),
            "z_mean": float(z[m].mean()),
            "abs_z_mean": float(np.abs(z[m]).mean()),
            "R2_TS": mt["R2"], "R2_Main": mm["R2"],
            "dR2": mm["R2"] - mt["R2"],
            "QLIKE_TS": mt["QLIKE"], "QLIKE_Main": mm["QLIKE"],
            "QLIKE_improv_pct": 100.0 * (mt["QLIKE"] - mm["QLIKE"]) / mt["QLIKE"],
            "MSE_TS": mt["MSE"], "MSE_Main": mm["MSE"],
            "MSE_improv_pct": 100.0 * (mt["MSE"] - mm["MSE"]) / mt["MSE"],
        })
    df = pd.DataFrame(rows)

    # Overall (pooled) row for reference
    mt_all = compute_metrics(ts, ac)
    mm_all = compute_metrics(mn, ac)
    print("\nPooled (all deciles):")
    print(f"  R2:    TS {mt_all['R2']:.4f}  Main {mm_all['R2']:.4f}  "
          f"dR2 {mm_all['R2'] - mt_all['R2']:+.4f}")
    print(f"  QLIKE: TS {mt_all['QLIKE']:.5f}  Main {mm_all['QLIKE']:.5f}  "
          f"improv {100*(mt_all['QLIKE']-mm_all['QLIKE'])/mt_all['QLIKE']:+.2f}%")
    print(f"  MSE:   TS {mt_all['MSE']:.5f}  Main {mm_all['MSE']:.5f}  "
          f"improv {100*(mt_all['MSE']-mm_all['MSE'])/mt_all['MSE']:+.2f}%")

    pd.set_option("display.width", 200)
    print("\nPer z-score decile (z = forward return / sqrt(TS pred)):")
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    csv_path = out_dir / "zscore_decile_improvement.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nSaved: {csv_path}")

    # ---- Figure: 3 panels (dR2, QLIKE %, MSE %) by z decile ----------------
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titleweight": "bold", "axes.titlesize": 12})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    x = np.arange(len(df))
    labels = [f"D{int(r.decile)}\nz̄={r.z_mean:+.2f}" for r in df.itertuples(index=False)]

    panels = [
        ("dR2", r"$\Delta R^2$  (Main $-$ TS)", "dR2"),
        ("QLIKE_improv_pct", "QLIKE improvement (%)", "QLIKE_improv_pct"),
        ("MSE_improv_pct", "MSE improvement (%)", "MSE_improv_pct"),
    ]
    for ax, (col, ylab, title) in zip(axes, panels):
        vals = df[col].values
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
        ax.bar(x, vals, color=colors, edgecolor="none", alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("z-score decile (signed)")
        ax.set_ylabel(ylab + "   ( + = Main better )")
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
        span = (vals.max() - vals.min()) or 1.0
        for xi, v in enumerate(vals):
            ax.text(xi, v + (0.02 * span if v >= 0 else -0.02 * span),
                    f"{v:+.3f}" if col == "dR2" else f"{v:+.1f}%",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=7.5, color="#333")

    fig.suptitle("BTC 4h v3: Main Model vs TS-only by z-score decile "
                 f"(pooled OOS, N={len(z):,})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = out_dir / "zscore_decile_improvement.png"
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
