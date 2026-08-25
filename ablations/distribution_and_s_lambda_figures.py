"""Two paper figures:

1. Distribution of vol predictions — TS-only vs Main Model vs Realized,
   pooled across all 6 production windows.

2. (s, λ) joint distribution — does high signal magnitude correlate with
   fast or slow decay?  Hex-binned density of all articles' per-article
   (s, λ) from the W6 main-model head, with category centroids overlaid.

Outputs:
  results/analysis/btc_news_impact_v3/prediction_distribution.png
  results/analysis/btc_news_impact_v3/s_lambda_jointplot.png
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from main_model_cryptobert import CryptoBertMainModel

CB_PATH  = REPO_ROOT / "data/processed/btc_news_embeddings_cryptobert.npz"
LAB_PATH = REPO_ROOT / "data/processed/btc_news_categories_clean.npz"

# Defaults; overridable via --main-dir / --out-dir (see main()).
MAIN_DIR = REPO_ROOT / "results/main/btc_cryptobert_v2"
RUN_CFG  = MAIN_DIR / "run_config.json"
CKPT_DIR = MAIN_DIR / "checkpoints"
PRED_DIR = MAIN_DIR / "predictions"
OUT_DIR  = REPO_ROOT / "results/analysis/btc_news_impact_v3"

PY = 365 * 24 / 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Figure 1: Prediction distribution ──────────────────────────────────────
def fig_prediction_distribution():
    ts_all, mn_all, ac_all = [], [], []
    for f in sorted(PRED_DIR.glob("window_*_4h_predictions.npz")):
        d = np.load(f, allow_pickle=True)
        ts_all.append(d["ts_mlp"])
        mn_all.append(d["main_model"])
        ac_all.append(d["actual"])
    ts = np.concatenate(ts_all).astype(np.float64)
    mn = np.concatenate(mn_all).astype(np.float64)
    ac = np.concatenate(ac_all).astype(np.float64)
    # Annualized vol (%)
    ts_vol = np.sqrt(np.clip(ts, 1e-15, None) * PY) * 100
    mn_vol = np.sqrt(np.clip(mn, 1e-15, None) * PY) * 100
    ac_vol = np.sqrt(np.clip(ac, 1e-15, None) * PY) * 100

    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.titleweight": "bold", "axes.titlesize": 13})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Left: KDE/histogram overlaid
    bins = np.linspace(0, 200, 80)
    for arr, label, color, ls in [
        (ts_vol, "TS-only (no news)",  "#6B7280", "--"),
        (mn_vol, "Main Model (with news)", "#C0392B", "-"),
        (ac_vol, "Realized",           "black",   "-"),
    ]:
        h, edges = np.histogram(arr, bins=bins, density=True)
        ctr = 0.5*(edges[1:]+edges[:-1])
        axL.plot(ctr, h, color=color, linewidth=2.2, linestyle=ls, label=label)
        axL.fill_between(ctr, 0, h, color=color, alpha=0.10)
    axL.set_xlim(0, 200)
    axL.set_xlabel("Annualized volatility (%)")
    axL.set_ylabel("Density")
    axL.set_title(f"Distribution of 4h vol forecasts and realized vol  (N = {len(ac_vol):,})")
    axL.legend(loc="upper right", frameon=False)

    # Right: Forecast vs realized scatter (45-deg line)
    rng = (0, 200)
    axR.scatter(ac_vol, ts_vol, s=3, alpha=0.15, color="#6B7280",
                label="TS-only")
    axR.scatter(ac_vol, mn_vol, s=3, alpha=0.18, color="#C0392B",
                label="Main Model")
    axR.plot([0, 200], [0, 200], color="black", linewidth=1.0,
             linestyle=":", label="perfect calibration")
    axR.set_xlim(*rng); axR.set_ylim(*rng)
    axR.set_xlabel("Realized vol (%)")
    axR.set_ylabel("Predicted vol (%)")
    axR.set_title("Forecast vs realized (45° = perfect)")
    axR.legend(loc="upper left", frameon=False, markerscale=3)

    plt.tight_layout()
    out = OUT_DIR / "prediction_distribution.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─── Figure 2: (s, λ) joint distribution ────────────────────────────────────
def fig_s_lambda():
    cfg = json.loads(RUN_CFG.read_text())
    feat_dim = int(cfg.get("cryptobert_input_dim", 768))
    hidden = tuple(cfg["hidden"])

    cb = np.load(CB_PATH)
    si = np.argsort(cb["timestamps"])
    emb = cb["embeddings"][si].astype(np.float32)
    n = len(emb)

    lab = np.load(LAB_PATH, allow_pickle=True)
    order = np.argsort(lab["unix_timestamp"])
    cats = np.array(lab["category"][order], dtype=object)

    # Pool (s, λ) across all 6 windows for stability
    s_all  = np.zeros(n, dtype=np.float64)
    lam_all = np.zeros(n, dtype=np.float64)
    n_windows = 0
    for w in range(1, 7):
        ckpt = CKPT_DIR / f"main_model_w{w}_4h.pt"
        if not ckpt.exists(): continue
        m = CryptoBertMainModel(feat_dim, hidden=hidden).to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        m.eval()
        with torch.no_grad():
            B = 4096
            for i in range(0, n, B):
                x = torch.tensor(emb[i:i+B], dtype=torch.float32, device=DEVICE)
                out = m.news_head(m.input_norm(x))
                s_all[i:i+B]  += torch.nn.functional.softplus(out[:, 0]).cpu().numpy()
                lam_all[i:i+B] += (0.001 + 0.499 * torch.sigmoid(out[:, 1])).cpu().numpy()
        n_windows += 1
        del m
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
    s_all /= n_windows
    lam_all /= n_windows
    half_life_min = np.log(2) / np.clip(lam_all, 1e-6, None)
    print(f"Pooled (s, λ) over {n_windows} windows; n_articles = {n:,}")
    print(f"  s:  min={s_all.min():.3f}, median={np.median(s_all):.3f}, max={s_all.max():.3f}")
    print(f"  λ:  min={lam_all.min():.4f}, median={np.median(lam_all):.4f}, max={lam_all.max():.4f}")
    print(f"  Spearman ρ(s, λ) = {pd.Series(s_all).corr(pd.Series(lam_all), method='spearman'):.3f}")
    print(f"  Pearson  ρ(s, λ) = {pd.Series(s_all).corr(pd.Series(lam_all), method='pearson'):.3f}")

    # Plot
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.titleweight": "bold", "axes.titlesize": 13})

    fig = plt.figure(figsize=(10.5, 9.0))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           wspace=0.04, hspace=0.04)
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_topx  = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_righty = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Scatter (semi-transparent so 25k points stay legible)
    ax_main.scatter(s_all, lam_all, s=4, color="#374151", alpha=0.20,
                    linewidths=0, rasterized=True)

    ax_main.set_xlabel(r"Signal magnitude $s$")
    ax_main.set_ylabel(r"Decay rate $\lambda$  (per minute)")
    ax_main.set_xlim(0, np.percentile(s_all, 99.7))
    ax_main.set_ylim(0, np.percentile(lam_all, 99.7))

    # Marginal: top (s)
    ax_topx.hist(s_all, bins=80, color="#4B5563", alpha=0.85)
    ax_topx.set_ylabel("count")
    ax_topx.tick_params(labelbottom=False)
    ax_topx.spines["top"].set_visible(False)
    ax_topx.spines["right"].set_visible(False)

    # Marginal: right (λ)
    ax_righty.hist(lam_all, bins=80, orientation="horizontal",
                   color="#4B5563", alpha=0.85)
    ax_righty.set_xlabel("count")
    ax_righty.tick_params(labelleft=False)
    ax_righty.spines["top"].set_visible(False)
    ax_righty.spines["right"].set_visible(False)

    # Annotate the quadrants
    xmax = ax_main.get_xlim()[1]
    ymax = ax_main.get_ylim()[1]
    ax_main.text(xmax*0.97, ymax*0.04,
                 "High signal,\nslow decay",
                 ha="right", va="bottom", fontsize=10,
                 color="#7F1D1D", fontstyle="italic",
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor="#FEF2F2", edgecolor="#FCA5A5", alpha=0.85))
    ax_main.text(xmax*0.03, ymax*0.96,
                 "Low signal,\nfast decay",
                 ha="left", va="top", fontsize=10,
                 color="#1F2937", fontstyle="italic",
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor="#F3F4F6", edgecolor="#D1D5DB", alpha=0.85))

    fig.suptitle(r"Joint distribution of per-article $s$ and $\lambda$",
                 fontsize=13, y=0.995, fontweight="bold")

    out = OUT_DIR / "s_lambda_jointplot.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    import argparse
    global MAIN_DIR, RUN_CFG, CKPT_DIR, PRED_DIR, OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dir", default=str(MAIN_DIR),
                        help="Run dir with run_config.json, checkpoints/, predictions/.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    MAIN_DIR = Path(args.main_dir)
    RUN_CFG  = MAIN_DIR / "run_config.json"
    CKPT_DIR = MAIN_DIR / "checkpoints"
    PRED_DIR = MAIN_DIR / "predictions"
    OUT_DIR  = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Main dir: {MAIN_DIR}")
    print(f"Out dir:  {OUT_DIR}")
    fig_prediction_distribution()
    fig_s_lambda()


if __name__ == "__main__":
    main()
