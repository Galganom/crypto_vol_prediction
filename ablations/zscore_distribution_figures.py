"""Two figures showing the z-score bucket distributions for
{BTC, ETH} x {1h, 4h}:

  1. zscore_distribution_all.png — full distribution (-3 .. +3), log y.
  2. zscore_distribution_tails.png — tails only (|z| > 2), linear y.

Reads results/analysis/btc_news_impact_v3/zscore_buckets_all.csv produced by
zscore_test_all.py.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
IN_CSV = REPO_ROOT / "results/analysis/btc_news_impact_v3/zscore_buckets_all.csv"
OUT_DIR = REPO_ROOT / "results/analysis/btc_news_impact_v3"

CONFIGS = [("BTC", "1h"), ("BTC", "4h"), ("ETH", "1h"), ("ETH", "4h")]
PALETTE = {"Expected": "#9CA3AF", "TS": "#374151", "Main": "#C0392B"}


def load():
    df = pd.read_csv(IN_CSV)
    return df


def plot_full(df, out_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titleweight": "bold", "axes.titlesize": 12})
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True)
    for ax, (asset, h) in zip(axes.ravel(), CONFIGS):
        sub = df[(df["asset"] == asset) & (df["horizon"] == h)].reset_index(drop=True)
        x = np.arange(len(sub))
        w = 0.27
        ax.bar(x - w, sub["Expected %"], w, color=PALETTE["Expected"],
               edgecolor="black", linewidth=0.4, label="Expected N(0,1)")
        ax.bar(x,     sub["TS %"],       w, color=PALETTE["TS"],
               edgecolor="black", linewidth=0.4, label="TS-only")
        ax.bar(x + w, sub["Main %"],     w, color=PALETTE["Main"],
               edgecolor="black", linewidth=0.4, label="Main Model")
        ax.set_yscale("log")
        ax.set_ylim(0.05, 35)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["bin"].values, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"{asset}  {h}   (N = {int(sub['TS N'].sum()):,})")
        ax.set_ylabel("Fraction of bars (%)")
        ax.grid(axis="y", which="both", linewidth=0.4, alpha=0.6)
        ax.grid(axis="x", linewidth=0)
        if ax is axes[0, 0]:
            ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Z-score distribution (full range, 0.5σ bins)",
                 fontsize=14, fontweight="bold", y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_tails(df, out_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titleweight": "bold", "axes.titlesize": 12})
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True)
    tail_bins = ["< -3.0", "[-3.0, -2.5)", "[-2.5, -2.0)",
                 "[2.0, 2.5)", "[2.5, 3.0)", "≥ 3.0"]
    for ax, (asset, h) in zip(axes.ravel(), CONFIGS):
        sub = df[(df["asset"] == asset) & (df["horizon"] == h)
                 & (df["bin"].isin(tail_bins))].copy()
        sub = sub.set_index("bin").loc[tail_bins].reset_index()
        x = np.arange(len(sub))
        w = 0.27
        ax.bar(x - w, sub["Expected %"], w, color=PALETTE["Expected"],
               edgecolor="black", linewidth=0.5, label="Expected N(0,1)")
        bars_ts = ax.bar(x, sub["TS %"], w, color=PALETTE["TS"],
                         edgecolor="black", linewidth=0.5, label="TS-only")
        bars_mn = ax.bar(x + w, sub["Main %"], w, color=PALETTE["Main"],
                         edgecolor="black", linewidth=0.5, label="Main Model")
        for bs, vals in [(bars_ts, sub["TS %"].values),
                         (bars_mn, sub["Main %"].values)]:
            for b, v in zip(bs, vals):
                ax.text(b.get_x() + b.get_width()/2, v + 0.04, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8.5)
        for b, v in zip(ax.containers[0], sub["Expected %"].values):
            ax.text(b.get_x() + b.get_width()/2, v + 0.04, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8.5, color="#444")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["bin"].values, rotation=20, ha="right", fontsize=10)
        ax.set_title(f"{asset}  {h}   (N = {int(sub['TS N'].sum() + (df[(df['asset']==asset) & (df['horizon']==h) & (~df['bin'].isin(tail_bins))]['TS N'].sum())):,})")
        ax.set_ylabel("Fraction of bars (%)")
        ax.set_ylim(0, max(sub["TS %"].max(), sub["Main %"].max()) * 1.18)
        ax.grid(axis="y", linewidth=0.4, alpha=0.6)
        ax.grid(axis="x", linewidth=0)
        if ax is axes[0, 0]:
            ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Z-score tail distribution (|z| > 2)",
                 fontsize=14, fontweight="bold", y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    df = load()
    plot_full(df, OUT_DIR / "zscore_distribution_all.png")
    plot_tails(df, OUT_DIR / "zscore_distribution_tails.png")


if __name__ == "__main__":
    main()
