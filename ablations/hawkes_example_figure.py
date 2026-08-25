"""Figure: Hawkes-style news decomposition of a single high-impact vol forecast bar.

Picks the BTC 4h bar that follows the U.S. CPI release on 2025-10-24, when the
main model dramatically beat the TS-only baseline. Inside the 12-hour lookback,
each non-reactive article gets a (s, λ) from the trained news head; the per-bar
prediction shift is the result of the Hawkes-style aggregation

    I(τ) = Σ_i  s_i · 1[t_i ≤ τ] · exp(-λ_i · (τ - t_i))

The figure shows:
  Panel A — every article in the lookback as a marker on the timeline, colored
            by category, sized by s.
  Panel B — per-article kernel s_i · exp(-λ_i · age) plotted from its arrival
            time to the forecast bar; the dominant Macro article in red,
            everything else in gray.
  Panel C — the aggregate Hawkes intensity I(τ), with horizontal reference lines
            for the TS-only vs Main forecasts at bar time.

The contrast figure (`Bitcoin Holds $113K`-style commentary bar) uses the same
layout to show how a high-volume, low-signal stream looks.

Output: results/analysis/btc_news_impact_v3/hawkes_example_cpi_oct2025.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from main_model_common import (build_price_bars, align_bars_with_news,
                                load_news_features)
from main_model_cryptobert import CryptoBertMainModel

CB_PATH = REPO_ROOT / "data/processed/btc_news_embeddings_cryptobert.npz"
LAB_PATH = REPO_ROOT / "data/processed/btc_news_categories_clean.npz"
CKPT_PATH = REPO_ROOT / "results/main/btc_cryptobert_v2/checkpoints/main_model_w6_4h.pt"
PRED_PATH = REPO_ROOT / "results/main/btc_cryptobert_v2/predictions/window_6_4h_predictions.npz"
RUN_CFG = REPO_ROOT / "results/main/btc_cryptobert_v2/run_config.json"
OUT_DIR = REPO_ROOT / "results/analysis/btc_news_impact_v3"

LOOKBACK_H = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CAT_COLORS = {
    "Macro":                   "#DC2626",
    "Geopolitical":            "#7C3AED",
    "Regulatory_Legal":        "#0EA5E9",
    "Security_Crime":          "#F97316",
    "Institutional_Flows":     "#10B981",
    "Protocol_Infrastructure": "#3B82F6",
    "OnChain_Flows":           "#06B6D4",
    "Product_Adoption":        "#84CC16",
    "Market_Commentary":       "#94A3B8",
    "Opinion_Editorial":       "#A78BFA",
    "Newsletter_Series":       "#FBBF24",
    "Other":                   "#9CA3AF",
}


def load_model():
    import json
    cfg = json.loads(RUN_CFG.read_text())
    feat_dim = int(cfg.get("cryptobert_input_dim", 768))
    hidden = tuple(cfg["hidden"])
    m = CryptoBertMainModel(feat_dim, hidden=hidden).to(DEVICE)
    m.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True))
    m.eval()
    return m


def per_article_s_lambda(model, embeddings):
    s = np.empty(len(embeddings), dtype=np.float32)
    lam = np.empty(len(embeddings), dtype=np.float32)
    B = 4096
    with torch.no_grad():
        for i in range(0, len(embeddings), B):
            x = torch.tensor(embeddings[i:i+B], dtype=torch.float32, device=DEVICE)
            xn = model.input_norm(x)
            out = model.news_head(xn)
            s[i:i+B] = torch.nn.functional.softplus(out[:, 0]).cpu().numpy()
            lam[i:i+B] = (0.001 + 0.499 * torch.sigmoid(out[:, 1])).cpu().numpy()
    return s, lam


def make_figure(bar_dt_utc, title_suffix, fig_path, dominant_cat="Macro"):
    """bar_dt_utc: pandas.Timestamp UTC of the forecast bar."""

    # ── Load corpus, labels, model
    cb = np.load(CB_PATH)
    si = np.argsort(cb["timestamps"])
    emb = cb["embeddings"][si].astype(np.float32)
    art_ts = cb["timestamps"][si].astype(np.int64)
    lab = np.load(LAB_PATH, allow_pickle=True)
    order = np.argsort(lab["unix_timestamp"])
    cats = np.array(lab["category"][order], dtype=object)
    nd = pd.read_parquet(REPO_ROOT / "data/news/processed_news.parquet")
    btc = nd[nd["asset"]=="BTC"].sort_values("unix_timestamp").reset_index(drop=True)
    titles_all = btc["title"].astype(str).values

    news = load_news_features("BTC")
    nonreactive = news["nonreactive_mask"].astype(bool)

    model = load_model()
    s_all, lam_all = per_article_s_lambda(model, emb)

    bar_ts = int(bar_dt_utc.timestamp())
    lo, hi = bar_ts - LOOKBACK_H*3600, bar_ts
    in_window = (art_ts >= lo) & (art_ts < hi) & nonreactive
    idx = np.where(in_window)[0]
    if len(idx) == 0:
        raise RuntimeError(f"No non-reactive articles in {LOOKBACK_H}h lookback of {bar_dt_utc}")
    t_i = art_ts[idx]
    s_i = s_all[idx]
    lam_i = lam_all[idx]
    cat_i = cats[idx]
    title_i = titles_all[idx]
    age_i_min = (bar_ts - t_i) / 60.0
    contrib_at_bar = s_i * np.exp(-lam_i * age_i_min)
    order_by_contrib = np.argsort(-contrib_at_bar)
    print(f"\n=== Bar {bar_dt_utc} — {len(idx)} non-reactive articles in lookback ===")
    for rank, k in enumerate(order_by_contrib[:6], 1):
        print(f"  {rank}. [{cat_i[k]:18s}]  s={s_i[k]:5.2f}  λ={lam_i[k]:.3f}  age={age_i_min[k]:6.1f} min  contrib@bar={contrib_at_bar[k]:5.2f}  {title_i[k][:75]}")

    # Identify dominant article(s)
    dom_mask = (cat_i == dominant_cat)
    if not dom_mask.any():
        print(f"  No {dominant_cat} article in window — falling back to top-contrib article")
        dom_mask = np.zeros_like(dom_mask)
        dom_mask[order_by_contrib[0]] = True

    # Aggregate Hawkes intensity over the 12h window (1-min resolution)
    grid_min = 1
    tau_min = np.arange(-LOOKBACK_H*60, 1, grid_min)  # minutes relative to bar
    tau_abs = bar_ts + tau_min*60
    intensity = np.zeros_like(tau_min, dtype=np.float64)
    intensity_dom = np.zeros_like(intensity)
    intensity_other = np.zeros_like(intensity)
    for k in range(len(idx)):
        active = tau_abs >= t_i[k]
        if not active.any(): continue
        age_local = (tau_abs - t_i[k]) / 60.0
        contrib_k = s_i[k] * np.exp(-lam_i[k] * np.clip(age_local, 0, None))
        contrib_k[~active] = 0.0
        intensity += contrib_k
        if dom_mask[k]:
            intensity_dom += contrib_k
        else:
            intensity_other += contrib_k

    # ── Load predictions
    pd_pred = np.load(PRED_PATH, allow_pickle=True)
    pred_dt = pd.to_datetime(pd_pred["datetime"], utc=True)
    pred_ts = pred_dt.astype("int64") // 10**9
    bar_idx = int(np.argmin(np.abs(pred_ts - bar_ts)))
    found = pd.Timestamp(pred_ts.values[bar_idx], unit="s", tz="UTC")
    if abs((found - bar_dt_utc).total_seconds()) > 7200:
        raise RuntimeError(f"closest test bar {found} too far from {bar_dt_utc}")
    ts_pred = float(pd_pred["ts_mlp"][bar_idx])
    main_pred = float(pd_pred["main_model"][bar_idx])
    actual = float(pd_pred["actual"][bar_idx])

    PY = 365 * 24 / 4
    ts_vol = float(np.sqrt(max(ts_pred, 1e-15)*PY))
    main_vol = float(np.sqrt(max(main_pred, 1e-15)*PY))
    act_vol = float(np.sqrt(max(actual, 1e-15)*PY))

    # ── Plot
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titleweight": "bold"})
    fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                                         gridspec_kw=dict(height_ratios=[0.7, 2, 1.2]))

    tau_dt = pd.to_datetime(tau_abs, unit="s", utc=True)

    # Panel A: article markers
    for cat, color in CAT_COLORS.items():
        mask = cat_i == cat
        if mask.any():
            axA.scatter(pd.to_datetime(t_i[mask], unit="s", utc=True),
                        np.zeros(mask.sum()), s=20+30*s_i[mask], c=color,
                        edgecolor="black", linewidth=0.4, alpha=0.85,
                        label=cat.replace("_", " "))
    axA.set_yticks([])
    axA.set_ylim(-0.5, 0.5)
    axA.set_title(f"Articles in 12h lookback (marker size ∝ $s_i$)  —  {len(idx)} non-reactive items")
    axA.legend(loc="upper left", ncol=4, fontsize=7.5, frameon=True)

    # Panel B: per-article decay curves
    for k in order_by_contrib[::-1]:  # plot dominant last so it's on top
        active = tau_abs >= t_i[k]
        age_local = (tau_abs - t_i[k]) / 60.0
        c_k = s_i[k] * np.exp(-lam_i[k] * np.clip(age_local, 0, None))
        c_k[~active] = np.nan
        if dom_mask[k]:
            axB.plot(tau_dt, c_k, color=CAT_COLORS.get(dominant_cat, "#DC2626"),
                     linewidth=2.5, zorder=10,
                     label=f"{dominant_cat.replace('_', ' ')}: " + (title_i[k][:60] + "…" if len(title_i[k])>60 else title_i[k]))
        else:
            axB.plot(tau_dt, c_k, color=CAT_COLORS.get(cat_i[k], "#9CA3AF"),
                     linewidth=0.9, alpha=0.55, zorder=1)
    axB.set_ylabel(r"Per-article intensity  $s_i \cdot e^{-\lambda_i (t-t_i)}$")
    axB.set_title("Per-article Hawkes kernels  —  $s_i$ at arrival, exponential decay at rate $\\lambda_i$")
    axB.legend(loc="upper left", fontsize=8.5, frameon=True)

    # Panel C: aggregate intensity + forecast lines
    axC.fill_between(tau_dt, 0, intensity_other, color="#9CA3AF", alpha=0.40,
                     label="Other articles' contribution")
    axC.fill_between(tau_dt, intensity_other, intensity_other+intensity_dom,
                     color=CAT_COLORS.get(dominant_cat, "#DC2626"), alpha=0.55,
                     label=f"{dominant_cat.replace('_', ' ')} contribution")
    axC.plot(tau_dt, intensity, color="black", linewidth=1.3, zorder=10)
    # Forecast reference (annotation only — Hawkes intensity is unitless wrt vol;
    # show the model's vol-shift implied by news as text)
    log_shift = np.log(max(main_pred, 1e-15)) - np.log(max(ts_pred, 1e-15))
    ax_pct_shift = (np.exp(log_shift/2) - 1) * 100   # % shift in vol
    axC.set_ylabel(r"Aggregate intensity $I(\tau)=\sum_i s_i e^{-\lambda_i (\tau-t_i)}$")
    axC.set_title(f"Aggregate news intensity   —   model's vol forecast shift at bar: "
                  f"TS-only $\\hat\\sigma$ = {ts_vol*100:.1f}%, "
                  f"Main = {main_vol*100:.1f}% (Δ = {ax_pct_shift:+.1f}%); "
                  f"realized $\\hat\\sigma$ = {act_vol*100:.1f}%")
    axC.legend(loc="upper left", fontsize=8.5, frameon=True)
    axC.axvline(pd.to_datetime(bar_ts, unit="s", utc=True),
                color="black", linestyle="--", linewidth=0.8, alpha=0.7)

    axC.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axC.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axC.set_xlabel(f"Hour of {bar_dt_utc.strftime('%Y-%m-%d')} (UTC)  —  4h forecast bar marked at right edge")

    plt.suptitle(f"Hawkes-style news decomposition at BTC 4h bar  {bar_dt_utc}  {title_suffix}",
                 y=0.998, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


def main():
    # 1) Macro-dominated bar: U.S. CPI release Oct 24 2025 (CPI at 12:30 UTC)
    make_figure(pd.Timestamp("2025-10-24 13:00:00", tz="UTC"),
                title_suffix="(U.S. September CPI release)",
                fig_path=OUT_DIR / "hawkes_example_cpi_oct2025.png",
                dominant_cat="Macro")

    # 2) Jobs-dominated bar: July 3 2025 NFP release
    make_figure(pd.Timestamp("2025-07-03 13:00:00", tz="UTC"),
                title_suffix="(U.S. June Jobs Data release)",
                fig_path=OUT_DIR / "hawkes_example_nfp_jul2025.png",
                dominant_cat="Macro")

    # 3) Geopolitical example — Iran/Israel ceasefire June 2025
    try:
        make_figure(pd.Timestamp("2025-06-24 04:00:00", tz="UTC"),
                    title_suffix="(Iran/Israel ceasefire headline)",
                    fig_path=OUT_DIR / "hawkes_example_geopolitical_jun2025.png",
                    dominant_cat="Geopolitical")
    except Exception as e:
        print(f"  Skipped geopolitical example: {e}")


if __name__ == "__main__":
    main()
