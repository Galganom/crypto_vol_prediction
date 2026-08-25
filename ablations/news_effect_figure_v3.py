"""Paper-quality figure: time series of the forecast (top) and the decay
structure of the top-3 contributing articles (bottom).

Top    — TS-only vs Main vs Realized over ±24h, with the 3 article arrivals
         marked on the timeline.
Bottom — per-article decay kernels s_i · exp(-λ_i · age) for the top-3,
         each labeled with its (truncated) headline, s value, and half-life.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from main_model_common import load_news_features
from main_model_cryptobert import CryptoBertMainModel

CB_PATH = REPO_ROOT / "data/processed/btc_news_embeddings_cryptobert.npz"
RUN_CFG = REPO_ROOT / "results/main/btc_cryptobert_v2/run_config.json"
OUT_DIR = REPO_ROOT / "results/analysis/btc_news_impact_v3"

LOOKBACK_H = 12
PY = 365 * 24 / 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(window):
    cfg = json.loads(RUN_CFG.read_text())
    feat_dim = int(cfg.get("cryptobert_input_dim", 768))
    hidden = tuple(cfg["hidden"])
    ckpt = REPO_ROOT / f"results/main/btc_cryptobert_v2/checkpoints/main_model_w{window}_4h.pt"
    m = CryptoBertMainModel(feat_dim, hidden=hidden).to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    m.eval()
    return m


def per_article_s_lambda(model, embeddings):
    s = np.empty(len(embeddings), dtype=np.float32)
    lam = np.empty(len(embeddings), dtype=np.float32)
    with torch.no_grad():
        B = 4096
        for i in range(0, len(embeddings), B):
            x = torch.tensor(embeddings[i:i+B], dtype=torch.float32, device=DEVICE)
            out = model.news_head(model.input_norm(x))
            s[i:i+B] = torch.nn.functional.softplus(out[:, 0]).cpu().numpy()
            lam[i:i+B] = (0.001 + 0.499 * torch.sigmoid(out[:, 1])).cpu().numpy()
    return s, lam


def make(bar_dt_utc, window, fig_path,
         plot_hours_before=12, plot_hours_after=12, top_n=3):
    pred_path = REPO_ROOT / f"results/main/btc_cryptobert_v2/predictions/window_{window}_4h_predictions.npz"
    cb = np.load(CB_PATH)
    si = np.argsort(cb["timestamps"])
    emb = cb["embeddings"][si].astype(np.float32)
    art_ts = cb["timestamps"][si].astype(np.int64)
    nd = pd.read_parquet(REPO_ROOT / "data/news/processed_news.parquet")
    btc = nd[nd["asset"]=="BTC"].sort_values("unix_timestamp").reset_index(drop=True)
    titles_all = btc["title"].astype(str).values

    news = load_news_features("BTC")
    nonreactive = news["nonreactive_mask"].astype(bool)
    model = load_model(window)
    s_all, lam_all = per_article_s_lambda(model, emb)

    bar_ts = int(bar_dt_utc.timestamp())
    lo, hi = bar_ts - LOOKBACK_H*3600, bar_ts
    in_window = (art_ts >= lo) & (art_ts < hi) & nonreactive
    idx = np.where(in_window)[0]
    t_i = art_ts[idx]; s_i = s_all[idx]; lam_i = lam_all[idx]
    titles = titles_all[idx]
    age_min = (bar_ts - t_i) / 60.0
    contrib_at_bar = s_i * np.exp(-lam_i * age_min)
    half_lives = np.log(2) / np.clip(lam_i, 1e-6, None)
    rank = np.argsort(-contrib_at_bar)[:top_n]
    print(f"\n=== Bar {bar_dt_utc} (W{window}) — top {top_n} articles ===")
    for r, k in enumerate(rank, 1):
        print(f"  {r}. s={s_i[k]:5.2f}  t½={half_lives[k]:6.1f}m  age={age_min[k]:6.1f}m  contrib={contrib_at_bar[k]:5.2f}  | {titles[k][:75]}")

    # Predictions across ±plot window
    pred = np.load(pred_path, allow_pickle=True)
    pred_dt = pd.to_datetime(pred["datetime"], utc=True)
    pred_ts = (pred_dt.astype("int64") // 10**9).values
    plot_lo = bar_ts - plot_hours_before*3600
    plot_hi = bar_ts + plot_hours_after*3600
    pmask = (pred_ts >= plot_lo) & (pred_ts <= plot_hi)
    pdt = pred_dt[pmask]
    ts_p = pred["ts_mlp"][pmask].astype(np.float64)
    mn_p = pred["main_model"][pmask].astype(np.float64)
    ac_p = pred["actual"][pmask].astype(np.float64)
    ts_vol = np.sqrt(np.clip(ts_p, 1e-15, None) * PY) * 100
    mn_vol = np.sqrt(np.clip(mn_p, 1e-15, None) * PY) * 100
    ac_vol = np.sqrt(np.clip(ac_p, 1e-15, None) * PY) * 100
    bidx = int(np.argmin(np.abs(pred_ts[pmask] - bar_ts)))

    # ── Figure
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.titleweight": "bold", "axes.titlesize": 13,
        "axes.labelsize": 12, "legend.fontsize": 11,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
    })
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1.0, 1.05],
                                                    hspace=0.18))

    # ── Top: forecasts
    axT.plot(pdt, ts_vol, color="#6B7280", linewidth=2.0, linestyle="--",
             label="Forecast without news")
    axT.plot(pdt, mn_vol, color="#C0392B", linewidth=2.6,
             label="Forecast with news")
    axT.plot(pdt, ac_vol, color="black", linewidth=1.6, alpha=0.6,
             label="Realized")
    axT.axvline(pd.to_datetime(bar_ts, unit="s", utc=True),
                color="black", linestyle=":", linewidth=0.9, alpha=0.45)

    # Mark the 3 article arrivals on the top panel with numbered ticks
    arr_color = ["#7F1D1D", "#B91C1C", "#EF4444"]
    for r_pos, k in enumerate(rank):
        arr_dt = pd.to_datetime(t_i[k], unit="s", utc=True)
        axT.axvline(arr_dt, color=arr_color[r_pos], linewidth=1.2,
                    alpha=0.55, linestyle="-")
        axT.annotate(f"{r_pos+1}", xy=(arr_dt, axT.get_ylim()[1]*0.95),
                     ha="center", va="top", fontsize=11, fontweight="bold",
                     color=arr_color[r_pos],
                     bbox=dict(boxstyle="circle,pad=0.18",
                               facecolor="white", edgecolor=arr_color[r_pos],
                               linewidth=1.3))

    axT.set_ylabel("Annualized volatility (%)")
    axT.set_title("Vol forecast — with vs without news")
    axT.legend(loc="upper left", frameon=False)

    # ── Bottom: stacked decay contributions of top 3, extending past the bar.
    tau_abs = np.arange(plot_lo, plot_hi+60, 60).astype(np.int64)
    tau_dt = pd.to_datetime(tau_abs, unit="s", utc=True)
    kernels = np.zeros((len(rank), len(tau_abs)), dtype=np.float64)
    labels = []
    for r_pos, k in enumerate(rank):
        active = tau_abs >= t_i[k]
        age_local = (tau_abs - t_i[k]) / 60.0
        ker = s_i[k] * np.exp(-lam_i[k] * np.clip(age_local, 0, None))
        ker[~active] = 0.0
        kernels[r_pos] = ker
        title_short = titles[k][:70] + ("…" if len(titles[k]) > 70 else "")
        labels.append(f"{r_pos+1}.  s={s_i[k]:.2f}, "
                      f"half-life={half_lives[k]:.0f} min  —  {title_short}")

    cumulative = np.zeros(len(tau_abs))
    for r_pos in range(len(rank)):
        new_top = cumulative + kernels[r_pos]
        axB.fill_between(tau_dt, cumulative, new_top,
                         color=arr_color[r_pos], alpha=0.55,
                         linewidth=0, label=labels[r_pos])
        axB.plot(tau_dt, new_top, color=arr_color[r_pos], linewidth=0.9,
                 alpha=0.9)
        cumulative = new_top
    axB.plot(tau_dt, cumulative, color="black", linewidth=2.0, alpha=0.9,
             label="Aggregate (sum of top 3)")

    axB.axvline(pd.to_datetime(bar_ts, unit="s", utc=True),
                color="black", linestyle=":", linewidth=0.9, alpha=0.45)
    axB.text(pd.to_datetime(bar_ts, unit="s", utc=True),
             axB.get_ylim()[1]*0.96, "  forecast bar",
             ha="left", va="top", fontsize=10, color="#555")

    axB.set_ylabel(r"Per-article intensity  $s_i \cdot e^{-\lambda_i (t-t_i)}$")
    axB.set_title("Decay structure of the top 3 articles in the 12h lookback")
    axB.legend(loc="upper left", frameon=True, fontsize=10)
    axB.set_xlim(pd.to_datetime(plot_lo, unit="s", utc=True),
                 pd.to_datetime(plot_hi, unit="s", utc=True))
    axB.set_xlabel(f"{bar_dt_utc.strftime('%Y-%m-%d')} (UTC)")
    axB.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    axB.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    fig.suptitle(f"BTC 4h vol forecast on {bar_dt_utc.strftime('%Y-%m-%d %H:%M')} UTC",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


def main():
    make(pd.Timestamp("2025-11-03 13:15:00", tz="UTC"), window=6,
         fig_path=OUT_DIR / "news_effect_v3_nov2025.png")
    make(pd.Timestamp("2024-03-05 18:30:00", tz="UTC"), window=3,
         fig_path=OUT_DIR / "news_effect_v3_mar2024.png")


if __name__ == "__main__":
    main()
