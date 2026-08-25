"""Analytical figures: s vs lambda relationship, which news mattered most.

Generates:
  fig_s_lambda_per_window.png    Per-(window, category) scatter of (λ, s) with
                                 correlation; shows how event-size and decay
                                 trade off. Bigger λ (faster decay) tends to
                                 pair with smaller s in some categories.
  fig_category_impact.png        4-panel breakdown by category — article volume,
                                 mean half-life, mean event size, and total
                                 impact share.
  fig_top_news_events.png        Top 20 most-impactful individual articles
                                 (by absolute multiplicative log-effect on RV)
                                 with titles and categories.

All saved to results/figures/.
"""
import os
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = REPO_ROOT / "results/analysis/btc_news_impact_4h"
OUT_DIR = REPO_ROOT / "results/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(context="paper", style="white", palette="deep")
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Category palette (semantic colors)
CAT_ORDER = [
    "Macro",
    "Policy/Legal",
    "ETF/Institutional/Adoption",
    "Exchange/Intermediary",
    "Protocol/Mining",
    "Market/Price/Technical",
    "Security/Crime",
    "Other",
]
CAT_COLORS = {
    "Macro":                      "#2563EB",  # blue
    "Policy/Legal":               "#7C3AED",  # violet
    "ETF/Institutional/Adoption": "#0EA5E9",  # cyan
    "Exchange/Intermediary":      "#F59E0B",  # amber
    "Protocol/Mining":            "#10B981",  # emerald
    "Market/Price/Technical":     "#EC4899",  # pink
    "Security/Crime":             "#DC2626",  # red
    "Other":                      "#6B7280",  # gray
}


# ═════════════════════════════════════════════════════════════════════════════
# Figure: s vs λ — per (window, category), with correlation analysis
# ═════════════════════════════════════════════════════════════════════════════
def fig_s_lambda_per_window():
    df = pd.read_csv(ANALYSIS / "final_window_category_s_lambda_stats.csv")
    fig = plt.figure(figsize=(13, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.4, 1.0, 1.0], wspace=0.35)

    # Panel A: scatter (λ, s) per (window, category)
    ax = fig.add_subplot(gs[0, 0])
    for cat in CAT_ORDER:
        sub = df[df["category"] == cat]
        ax.scatter(sub["lambda_weighted_mean_per_min"], sub["s_weighted_mean"],
                   s=70 + sub["unique_articles_in_test_lookback"] * 0.25,
                   color=CAT_COLORS[cat], alpha=0.85, edgecolors="white",
                   linewidth=0.7, label=cat, zorder=3)
        # Connect same category across windows
        ax.plot(sub.sort_values("window")["lambda_weighted_mean_per_min"],
                sub.sort_values("window")["s_weighted_mean"],
                color=CAT_COLORS[cat], alpha=0.2, linewidth=0.8, zorder=1)
    # Reference half-life lines on lambda axis
    for hl_min in [15, 30, 60, 120]:
        lam = np.log(2) / hl_min
        if df["lambda_weighted_mean_per_min"].min() <= lam <= df["lambda_weighted_mean_per_min"].max():
            ax.axvline(lam, color="#9CA3AF", linewidth=0.6, linestyle=":", alpha=0.7)
            ax.text(lam, ax.get_ylim()[1] if ax.get_ylim()[1] else df["s_weighted_mean"].max() + 0.05,
                    f"  {hl_min}m HL", fontsize=8, color="#6B7280", va="top")
    # Pearson correlation
    pr, p_p = pearsonr(df["lambda_weighted_mean_per_min"], df["s_weighted_mean"])
    sr, p_s = spearmanr(df["lambda_weighted_mean_per_min"], df["s_weighted_mean"])
    ax.text(0.02, 0.97,
            f"Pearson r = {pr:+.3f}  (p = {p_p:.2e})\nSpearman ρ = {sr:+.3f}  (p = {p_s:.2e})",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#9CA3AF", lw=0.5))
    ax.set_xlabel(r"Decay rate $\lambda$ (per minute)")
    ax.set_ylabel(r"Event size $s$ (weighted mean)")
    ax.set_title("Per-(window, category): event size vs. decay rate")
    ax.legend(loc="lower right", fontsize=8, ncol=2, columnspacing=0.6,
              handletextpad=0.4, markerscale=0.7)

    # Panel B: per-category Pearson correlation across windows
    ax2 = fig.add_subplot(gs[0, 1])
    rows = []
    for cat in CAT_ORDER:
        sub = df[df["category"] == cat]
        if len(sub) >= 3:
            r, p = pearsonr(sub["lambda_weighted_mean_per_min"], sub["s_weighted_mean"])
            rows.append({"category": cat, "r": r, "p": p, "n": len(sub)})
    cd = pd.DataFrame(rows).sort_values("r")
    bars = ax2.barh(cd["category"], cd["r"],
                    color=[CAT_COLORS[c] for c in cd["category"]],
                    edgecolor="white", linewidth=0.8)
    for bar, p, r in zip(bars, cd["p"], cd["r"]):
        star = " *" if p < 0.05 else ""
        ax2.text(r + (0.02 if r >= 0 else -0.02), bar.get_y() + bar.get_height()/2,
                 f"{r:+.2f}{star}", va="center",
                 ha="left" if r >= 0 else "right", fontsize=8, color="#1F2937")
    ax2.axvline(0, color="black", linewidth=0.6)
    ax2.set_xlim(-1.05, 1.05)
    ax2.set_xlabel("Pearson r (across windows)")
    ax2.set_title("s–λ correlation\nby category")

    # Panel C: distribution of half-lives by category (boxplot)
    ax3 = fig.add_subplot(gs[0, 2])
    cats_ordered_by_hl = (df.groupby("category")["half_life_weighted_median_min"]
                            .mean().sort_values().index.tolist())
    box_data = [df[df["category"] == c]["half_life_weighted_median_min"].values
                for c in cats_ordered_by_hl]
    bp = ax3.boxplot(box_data, vert=False, patch_artist=True,
                     widths=0.55, showmeans=True,
                     meanprops=dict(marker="D", markerfacecolor="white",
                                    markeredgecolor="black", markersize=4))
    for patch, c in zip(bp["boxes"], cats_ordered_by_hl):
        patch.set_facecolor(CAT_COLORS[c]); patch.set_alpha(0.75)
        patch.set_edgecolor("#374151")
    ax3.set_yticklabels(cats_ordered_by_hl, fontsize=8)
    ax3.set_xlabel("Half-life (minutes)")
    ax3.set_title("Per-category half-life\ndistribution (6 windows)")

    plt.tight_layout()
    out = OUT_DIR / "fig_s_lambda_per_window.png"
    plt.savefig(out); plt.close()
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure: per-category impact breakdown
# ═════════════════════════════════════════════════════════════════════════════
def fig_category_impact():
    stats = pd.read_csv(ANALYSIS / "final_category_s_lambda_stats.csv")
    impact = pd.read_csv(ANALYSIS / "final_category_impact_summary.csv")
    dist = pd.read_csv(ANALYSIS / "final_category_distribution.csv")
    df = stats.merge(impact, on="category").merge(dist, on="category")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))

    # Panel 1: article volume share
    ax = axes[0, 0]
    sub = df.sort_values("article_count", ascending=True)
    bars = ax.barh(sub["category"], sub["article_count"],
                   color=[CAT_COLORS[c] for c in sub["category"]],
                   edgecolor="white", linewidth=0.8)
    for bar, n, share in zip(bars, sub["article_count"], sub["article_share"]):
        ax.text(n + 80, bar.get_y() + bar.get_height()/2,
                f"{int(n):,} ({share*100:.1f}%)", va="center", fontsize=8, color="#1F2937")
    ax.set_xlabel("Articles in corpus (count)")
    ax.set_title("(a) Article volume by category")
    ax.set_xlim(0, sub["article_count"].max() * 1.18)

    # Panel 2: half-life
    ax = axes[0, 1]
    sub = df.sort_values("half_life_weighted_median_min", ascending=True)
    bars = ax.barh(sub["category"], sub["half_life_weighted_median_min"],
                   color=[CAT_COLORS[c] for c in sub["category"]],
                   edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, sub["half_life_weighted_median_min"]):
        ax.text(v + 1, bar.get_y() + bar.get_height()/2,
                f"{v:.1f} min", va="center", fontsize=8, color="#1F2937")
    ax.set_xlabel("Median half-life (minutes)")
    ax.set_title("(b) Decay duration — how long news matters")
    ax.set_xlim(0, sub["half_life_weighted_median_min"].max() * 1.20)

    # Panel 3: mean event size s
    ax = axes[1, 0]
    sub = df.sort_values("s_weighted_mean", ascending=True)
    bars = ax.barh(sub["category"], sub["s_weighted_mean"],
                   xerr=sub["s_weighted_std"], capsize=3, ecolor="#374151",
                   color=[CAT_COLORS[c] for c in sub["category"]],
                   edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, sub["s_weighted_mean"]):
        ax.text(v + 0.05, bar.get_y() + bar.get_height()/2,
                f"{v:.2f}", va="center", fontsize=8, color="#1F2937")
    ax.set_xlabel(r"Event size $s$ (weighted mean ± std)")
    ax.set_title("(c) Mean event size — per-article signal")
    ax.set_xlim(0, sub["s_weighted_mean"].max() * 1.30)

    # Panel 4: total absolute impact share
    ax = axes[1, 1]
    sub = df.sort_values("impact_share_abs", ascending=True)
    bars = ax.barh(sub["category"], sub["impact_share_abs"] * 100,
                   color=[CAT_COLORS[c] for c in sub["category"]],
                   edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, sub["impact_share_abs"]):
        ax.text(v * 100 + 0.4, bar.get_y() + bar.get_height()/2,
                f"{v*100:.1f}%", va="center", fontsize=8, color="#1F2937")
    ax.set_xlabel("Share of total absolute log-effect on RV (%)")
    ax.set_title("(d) Total impact share — what moved volatility most")
    ax.set_xlim(0, sub["impact_share_abs"].max() * 100 * 1.18)

    fig.suptitle("BTC 4h news impact decomposition — by category",
                 fontsize=12, fontweight="bold", y=1.00)
    plt.tight_layout()
    out = OUT_DIR / "fig_category_impact.png"
    plt.savefig(out); plt.close()
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure: top news events
# ═════════════════════════════════════════════════════════════════════════════
def fig_top_news_events(top_n=20):
    df = pd.read_csv(ANALYSIS / "final_top_event_exposures.csv")
    # An article appears multiple times (one row per bar it touches as the news
    # decays). Keep its single highest-impact bar exposure.
    df = (df.sort_values("abs_log_effect", ascending=False)
            .drop_duplicates("title", keep="first")
            .head(top_n)
            .reset_index(drop=True))
    df["title_short"] = df["title"].apply(lambda t: textwrap.shorten(str(t), width=72, placeholder="…"))
    df = df.iloc[::-1].reset_index(drop=True)  # so highest is at top

    fig, ax = plt.subplots(figsize=(13, 0.42 * top_n + 1.5))
    bars = ax.barh(np.arange(len(df)), df["signed_log_effect"],
                   color=[CAT_COLORS.get(c, "#6B7280") for c in df["category"]],
                   edgecolor="white", linewidth=0.6)
    for i, row in df.iterrows():
        ax.text(row["signed_log_effect"] + (0.005 if row["signed_log_effect"] >= 0 else -0.005),
                i, f' {row["title_short"]}',
                va="center", ha="left" if row["signed_log_effect"] >= 0 else "right",
                fontsize=8, color="#1F2937")
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks([])
    ax.set_xlabel("Signed log-multiplicative effect on RV at peak exposure")
    ax.set_title(f"Top {top_n} BTC 4h news events by absolute impact (out-of-sample, all 6 test windows)")
    # Build legend with categories that actually appear
    cats_used = [c for c in CAT_ORDER if c in df["category"].values]
    handles = [mpl.patches.Patch(facecolor=CAT_COLORS[c], label=c) for c in cats_used]
    ax.legend(handles=handles, loc="lower right", ncol=2, fontsize=8)

    # Stretch x-axis a bit for the right-side text labels
    xmin, xmax = df["signed_log_effect"].min(), df["signed_log_effect"].max()
    ax.set_xlim(xmin - 0.05, xmax + 0.55)
    plt.tight_layout()
    out = OUT_DIR / "fig_top_news_events.png"
    plt.savefig(out); plt.close()
    print(f"Saved: {out}")


def main():
    fig_s_lambda_per_window()
    fig_category_impact()
    fig_top_news_events(top_n=20)


if __name__ == "__main__":
    main()
