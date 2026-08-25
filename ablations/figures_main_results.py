"""Generate professional figures for the CryptoBERT-PCA volatility framework.

Produces:
  fig1_pooled_r2.png             Pooled R^2 across 5 models x {1h, 4h} x {BTC, ETH}
  fig2_per_window_r2.png         Per-window R^2 lines for each model (2x2 grid)
  fig3_dm_significance.png       DM t-stat heatmap (Main Model vs all on MSE/QLIKE)
  fig4_news_contribution.png     Delta R^2 of news-aware models over their baselines
  fig5_training_curves.png       Example P1+P2 train/val loss curves with phase markers
  fig6_pca_explained_variance.png Cumulative PCA explained variance per window
  fig7_feature_ablation.png      Sentiment vs CryptoBERT-PCA vs Raw CryptoBERT vs FinBERT

All saved at 300 DPI to results/figures/.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
OUT_DIR = RESULTS / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Global aesthetic ────────────────────────────────────────────────────────
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

MODELS = ["HAR", "EGARCH", "EGARCH+NED", "TS MLP", "Main Model"]
MODEL_COLORS = {
    "HAR":         "#9CA3AF",
    "EGARCH":      "#6B7280",
    "EGARCH+NED":  "#F59E0B",
    "TS MLP":      "#3B82F6",
    "Main Model":  "#DC2626",
}
ASSET_HORIZONS = [("BTC", "1h"), ("BTC", "4h"), ("ETH", "1h"), ("ETH", "4h")]
ASSET_DIRS = {"BTC": RESULTS / "main/btc_cryptobert", "ETH": RESULTS / "main/eth_cryptobert"}


def load_metrics(asset):
    """Per-window metrics + pooled means for one asset."""
    detail = pd.read_csv(ASSET_DIRS[asset] / "expanding_window_metrics.csv")
    means = pd.read_csv(ASSET_DIRS[asset] / "expanding_window_mean_metrics.csv")
    dm = pd.read_csv(ASSET_DIRS[asset] / "dm_test_results.csv")
    return detail, means, dm


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1 — Pooled R^2 across all configurations
# ═════════════════════════════════════════════════════════════════════════════
def fig1_pooled_r2():
    rows = []
    for asset in ["BTC", "ETH"]:
        _, means, _ = load_metrics(asset)
        for _, r in means.iterrows():
            rows.append({"asset": asset, "horizon": r["horizon"], "model": r["model"], "R2": r["R2"]})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, asset in zip(axes, ["BTC", "ETH"]):
        sub = df[df["asset"] == asset]
        pivot = sub.pivot(index="model", columns="horizon", values="R2").reindex(MODELS)
        x = np.arange(len(MODELS))
        w = 0.36
        bars1 = ax.bar(x - w/2, pivot["1h"], width=w, label="1h",
                       color=[MODEL_COLORS[m] for m in MODELS],
                       edgecolor="white", linewidth=0.6)
        bars2 = ax.bar(x + w/2, pivot["4h"], width=w, label="4h",
                       color=[MODEL_COLORS[m] for m in MODELS],
                       edgecolor="white", linewidth=0.6, alpha=0.55, hatch="//")
        for bars in (bars1, bars2):
            for bar in bars:
                v = bar.get_height()
                offset = 0.012 if v >= 0 else -0.022
                ax.text(bar.get_x() + bar.get_width()/2, v + offset, f"{v:.3f}",
                        ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=8, color="#1F2937")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x); ax.set_xticklabels(MODELS, rotation=18, ha="right")
        ax.set_title(f"{asset} — Pooled $R^2$ (6 expanding windows, N≈105k)")
        ax.set_ylabel("$R^2$" if asset == "BTC" else "")
        ax.set_ylim(min(df["R2"].min() - 0.04, -0.40), df["R2"].max() + 0.06)
    handles = [
        mpl.patches.Patch(facecolor="#888", edgecolor="white", label="1h horizon"),
        mpl.patches.Patch(facecolor="#888", edgecolor="white", alpha=0.55, hatch="//", label="4h horizon"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT_DIR / "fig1_pooled_r2.png"
    plt.savefig(out); plt.close()
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2 — Per-window R^2 trajectories
# ═════════════════════════════════════════════════════════════════════════════
def fig2_per_window_r2():
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True)
    for (asset, horizon), ax in zip(ASSET_HORIZONS, axes.flatten()):
        detail, _, _ = load_metrics(asset)
        sub = detail[detail["horizon"] == horizon].copy()
        for model in MODELS:
            m = sub[sub["model"] == model].sort_values("window")
            ax.plot(m["window"], m["R2"], marker="o", markersize=6,
                    linewidth=2 if model == "Main Model" else 1.4,
                    color=MODEL_COLORS[model], label=model,
                    zorder=4 if model == "Main Model" else 2)
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        ax.set_title(f"{asset} — {horizon} horizon")
        ax.set_xlabel("Expanding window (test period)" if axes.flatten().tolist().index(ax) >= 2 else "")
        ax.set_ylabel("$R^2$" if axes.flatten().tolist().index(ax) % 2 == 0 else "")
        ax.set_xticks(sorted(sub["window"].unique()))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.01))
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT_DIR / "fig2_per_window_r2.png"
    plt.savefig(out); plt.close()
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3 — DM significance heatmap (Main Model vs each on MSE & QLIKE)
# ═════════════════════════════════════════════════════════════════════════════
def fig3_dm_significance():
    rows = []
    for asset in ["BTC", "ETH"]:
        _, _, dm = load_metrics(asset)
        # DM t-stat: Main Model vs others (for vs_Main columns, model is the OTHER, so flip sign of t)
        # Use vs_TS_MLP rows (model='Main Model' has DM_MSE_t_vs_TS_MLP)
        means = dm[dm["window"].astype(str) == "mean"]
        for _, r in means.iterrows():
            if r["model"] == "Main Model":
                continue
            rows.append({
                "asset": asset, "horizon": r["horizon"], "model": r["model"],
                "metric": "MSE",
                "t": r.get("DM_MSE_t_vs_Main") if not pd.isna(r.get("DM_MSE_t_vs_Main")) else np.nan,
                "p": r.get("DM_MSE_p_vs_Main") if not pd.isna(r.get("DM_MSE_p_vs_Main")) else np.nan,
            })
            rows.append({
                "asset": asset, "horizon": r["horizon"], "model": r["model"],
                "metric": "QLIKE",
                "t": r.get("DM_QLIKE_t_vs_Main") if not pd.isna(r.get("DM_QLIKE_t_vs_Main")) else np.nan,
                "p": r.get("DM_QLIKE_p_vs_Main") if not pd.isna(r.get("DM_QLIKE_p_vs_Main")) else np.nan,
            })
    df = pd.DataFrame(rows)
    # Positive t in vs_Main row means OTHER worse than Main → keep that orientation: positive = Main wins.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, metric in zip(axes, ["MSE", "QLIKE"]):
        sub = df[df["metric"] == metric].copy()
        sub["row"] = sub["asset"] + " " + sub["horizon"]
        pivot_t = sub.pivot(index="row", columns="model", values="t")
        pivot_p = sub.pivot(index="row", columns="model", values="p")
        models_order = [m for m in MODELS if m != "Main Model"]
        pivot_t = pivot_t[models_order]
        pivot_p = pivot_p[models_order]
        annot = pivot_t.round(2).astype(str).copy()
        for r in pivot_p.index:
            for c in pivot_p.columns:
                t = pivot_t.loc[r, c]; p = pivot_p.loc[r, c]
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                annot.loc[r, c] = f"{t:+.2f}{star}"
        cmap = sns.diverging_palette(15, 230, s=85, l=55, as_cmap=True)
        sns.heatmap(pivot_t, ax=ax, cmap=cmap, center=0, vmin=-30, vmax=30,
                    annot=annot, fmt="", linewidths=0.5, linecolor="white",
                    cbar_kws={"label": f"DM t-stat ({metric}) — Main Model vs Other"})
        ax.set_title(f"DM test — {metric} (Main Model vs each)")
        ax.set_xlabel(""); ax.set_ylabel("")
    fig.text(0.5, -0.02,
             "Positive t-stat = Main Model loss is lower (better). Stars: *** p<0.001, ** p<0.01, * p<0.05.",
             ha="center", fontsize=9, color="#374151")
    plt.tight_layout()
    out = OUT_DIR / "fig3_dm_significance.png"
    plt.savefig(out); plt.close()
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4 — News contribution: ΔR² over corresponding base
# ═════════════════════════════════════════════════════════════════════════════
def fig4_news_contribution():
    """For each news-aware model, plot ΔR² vs its base:
       EGARCH+NED − EGARCH (news lift on EGARCH base)
       Main Model − TS MLP  (news lift on TS-MLP base)
    """
    rows = []
    for asset in ["BTC", "ETH"]:
        _, means, _ = load_metrics(asset)
        for horizon in ["1h", "4h"]:
            r2 = {r["model"]: r["R2"] for _, r in means[means["horizon"] == horizon].iterrows()}
            rows.append({"asset": asset, "horizon": horizon,
                         "pair": "EGARCH+NED − EGARCH",
                         "delta": r2["EGARCH+NED"] - r2["EGARCH"]})
            rows.append({"asset": asset, "horizon": horizon,
                         "pair": "Main Model − TS MLP",
                         "delta": r2["Main Model"] - r2["TS MLP"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4.3))
    df["label"] = df["asset"] + "\n" + df["horizon"]
    pivot = df.pivot(index="label", columns="pair", values="delta")
    pivot = pivot[["EGARCH+NED − EGARCH", "Main Model − TS MLP"]]
    x = np.arange(len(pivot.index))
    w = 0.36
    bars1 = ax.bar(x - w/2, pivot["EGARCH+NED − EGARCH"], width=w,
                   color="#F59E0B", edgecolor="white", linewidth=0.6,
                   label="EGARCH+NED − EGARCH")
    bars2 = ax.bar(x + w/2, pivot["Main Model − TS MLP"], width=w,
                   color="#DC2626", edgecolor="white", linewidth=0.6,
                   label="Main Model − TS MLP")
    for bars in (bars1, bars2):
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:+.3f}",
                    ha="center", va="bottom", fontsize=9, color="#1F2937")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Δ$R^2$ vs base")
    ax.set_title("News contribution to volatility forecasting")
    ax.legend(loc="upper right")
    plt.tight_layout()
    out = OUT_DIR / "fig4_news_contribution.png"
    plt.savefig(out); plt.close()
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 5 — Training curves (representative window)
# ═════════════════════════════════════════════════════════════════════════════
def fig5_training_curves():
    rep = ASSET_DIRS["BTC"] / "loss_curves" / "main_model_w3_4h_loss.csv"
    if not rep.exists():
        candidates = sorted((ASSET_DIRS["BTC"] / "loss_curves").glob("main_model_*_4h_loss.csv"))
        if not candidates:
            print("No loss curves found; skipping fig5.")
            return
        rep = candidates[len(candidates)//2]
    df = pd.read_csv(rep)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, phase in zip(axes, ["P1", "P2"]):
        sub = df[df["phase"] == phase]
        if sub.empty:
            ax.set_title(f"{phase} (no data)"); continue
        # Skip the first epoch's massive cold-start loss so the convergence
        # region is readable.
        sub = sub[sub["epoch"] >= 2] if phase == "P1" else sub
        ax.plot(sub["epoch"], sub["train_loss"], color="#3B82F6", linewidth=1.7,
                label="Train", alpha=0.85)
        ax.plot(sub["epoch"], sub["val_loss"], color="#DC2626", linewidth=2.0, label="Val")
        best_idx = sub["val_loss"].idxmin()
        best_ep = sub.loc[best_idx, "epoch"]
        best_v = sub.loc[best_idx, "val_loss"]
        ax.scatter([best_ep], [best_v], color="#DC2626", s=110, zorder=5,
                   marker="*", edgecolors="white", linewidth=1.2)
        ax.annotate(f"best val\nep={int(best_ep)}, {best_v:.4f}",
                    xy=(best_ep, best_v), xytext=(8, 12),
                    textcoords="offset points", fontsize=8, color="#7F1D1D",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#7F1D1D", lw=0.5))
        title = ("P1 — TS-only (news head frozen)" if phase == "P1"
                 else "P2 — news head trains, TS frozen")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Level loss (log-MSE)" if phase == "P1" else "")
        ax.legend(loc="upper right")
    fig.suptitle(f"Training curves — BTC W3 4h (representative)", fontsize=11, y=1.01)
    plt.tight_layout()
    out = OUT_DIR / "fig5_training_curves.png"
    plt.savefig(out); plt.close()
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 6 — PCA explained variance per window
# ═════════════════════════════════════════════════════════════════════════════
def fig6_pca_explained_variance():
    """Two-panel figure: (a) cumulative variance per window for both assets,
    (b) per-component variance to show the spectral decay."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    palette = sns.color_palette("viridis", n_colors=6)
    asset_styles = {"BTC": "-", "ETH": "--"}
    plotted_components = None

    # Left: cumulative variance, both assets, all 6 windows
    ax = axes[0]
    for asset in ["BTC", "ETH"]:
        meta_dir = ASSET_DIRS[asset] / "metadata"
        files = sorted(meta_dir.glob("cryptobert_pca_w*.npz"),
                       key=lambda p: int(p.stem.split("w")[-1]))
        for i, f in enumerate(files):
            d = np.load(f)
            cum = np.cumsum(d["explained_variance_ratio"])
            label = f"{asset} W{i+1}" if i in (0, len(files)-1) else None
            ax.plot(np.arange(1, len(cum) + 1), cum, color=palette[i],
                    linestyle=asset_styles[asset], linewidth=1.5, alpha=0.85,
                    label=label)
            plotted_components = len(cum)
    ax.axhline(0.85, color="#9CA3AF", linewidth=0.8, linestyle=":", alpha=0.8)
    ax.text(plotted_components, 0.85, " 85%", color="#6B7280", fontsize=9, va="bottom")
    ax.set_title("Cumulative explained variance — CryptoBERT PCA")
    ax.set_xlabel("Number of components"); ax.set_ylabel("Cumulative explained variance")
    handle_btc = mpl.lines.Line2D([], [], color="#374151", linestyle="-", label="BTC")
    handle_eth = mpl.lines.Line2D([], [], color="#374151", linestyle="--", label="ETH")
    handle_w = [mpl.lines.Line2D([], [], color=palette[i], linewidth=2.5, label=f"W{i+1}") for i in range(6)]
    leg1 = ax.legend(handles=[handle_btc, handle_eth], loc="upper left", title="Asset")
    ax.add_artist(leg1)
    ax.legend(handles=handle_w, loc="lower right", ncol=2, title="Window", fontsize=8)

    # Right: per-component (top 32) variance — shows where signal concentrates
    ax = axes[1]
    btc_files = sorted((ASSET_DIRS["BTC"] / "metadata").glob("cryptobert_pca_w*.npz"),
                        key=lambda p: int(p.stem.split("w")[-1]))
    K = 32
    for i, f in enumerate(btc_files):
        d = np.load(f)
        evr = d["explained_variance_ratio"][:K]
        ax.plot(np.arange(1, len(evr) + 1), evr, marker="o", markersize=3.5,
                color=palette[i], linewidth=1.4, alpha=0.85,
                label=f"W{i+1} (n={int(d['n_train_articles']):,})")
    ax.set_title(f"BTC — top {K} components: per-component variance share")
    ax.set_xlabel("Component index")
    ax.set_ylabel("Explained variance ratio")
    ax.set_xlim(0, K + 1)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out = OUT_DIR / "fig6_pca_explained_variance.png"
    plt.savefig(out); plt.close()
    print(f"Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 7 — Feature ablation (sentiment vs PCA vs raw vs FinBERT)
# ═════════════════════════════════════════════════════════════════════════════
def fig7_feature_ablation():
    p = RESULTS / "ablations/btc_all6_p2_feature_compare/p2_feature_mean_metrics.csv"
    if not p.exists():
        print(f"Skip fig7: {p} missing"); return
    df = pd.read_csv(p)
    # Map raw model names to display labels
    label_map = {
        "sentiment_decay":           "Sentiment (3-d)",
        "cryptobert_pca256_decay":   "CryptoBERT-PCA (256-d)",
        "cryptobert_raw_decay":      "CryptoBERT raw (768-d)",
        "finbert_raw_decay":         "FinBERT raw (768-d)",
    }
    df["label"] = df["model"].map(label_map)
    df = df.dropna(subset=["label"]).copy()
    df = df.sort_values("R2", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    palette = ["#3B82F6", "#F59E0B", "#10B981", "#DC2626"][:len(df)]

    # Left panel: R² bars
    ax = axes[0]
    bars = ax.barh(df["label"], df["R2"], color=palette, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, df["R2"]):
        ax.text(v + 0.0008, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=9, color="#1F2937")
    ax.set_xlabel("$R^2$")
    ax.set_title("BTC 4h — feature ablation (P2 only, 3 seeds, 6 windows pooled)")
    ax.set_xlim(df["R2"].min() - 0.001, df["R2"].max() + 0.005)

    # Right panel: QLIKE bars (lower is better)
    ax = axes[1]
    df_q = df.sort_values("QLIKE", ascending=False)
    bars = ax.barh(df_q["label"], df_q["QLIKE"], color=palette, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, df_q["QLIKE"]):
        ax.text(v + 0.0008, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=9, color="#1F2937")
    ax.set_xlabel("QLIKE (lower = better)")
    ax.set_title("BTC 4h — feature ablation (QLIKE)")
    ax.set_xlim(df_q["QLIKE"].min() - 0.002, df_q["QLIKE"].max() + 0.006)
    plt.tight_layout()
    out = OUT_DIR / "fig7_feature_ablation.png"
    plt.savefig(out); plt.close()
    print(f"Saved: {out}")


def main():
    print(f"Output directory: {OUT_DIR}")
    fig1_pooled_r2()
    fig2_per_window_r2()
    fig3_dm_significance()
    fig4_news_contribution()
    fig5_training_curves()
    fig6_pca_explained_variance()
    fig7_feature_ablation()
    print("\nAll figures generated.")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"  {f.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
