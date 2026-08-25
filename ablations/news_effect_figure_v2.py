"""Paper-quality figure: vol forecast WITHOUT vs WITH news, with a clean
horizontal-bar contribution chart of the news articles in the lookback.

Two panels:
  Top    — vol forecast over ±24h around the focal 4h bar.
           TS-only (no news) vs Main Model (with news) vs realized.
  Bottom — articles in the 12h lookback as horizontal bars, sorted by their
           contribution at the bar.  Each bar labeled with the full headline.

No category coloring; only the article text matters.

Outputs: results/analysis/btc_news_impact_v3/news_effect_<event>.png
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


def make(bar_dt_utc, window, fig_path, plot_hours_before=12, plot_hours_after=12,
         top_n_label=6):
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
    t_i = art_ts[idx]
    s_i = s_all[idx]
    lam_i = lam_all[idx]
    titles = titles_all[idx]
    age_min = (bar_ts - t_i) / 60.0
    contrib_at_bar = s_i * np.exp(-lam_i * age_min)
    half_lives = np.log(2) / np.clip(lam_i, 1e-6, None)
    rank = np.argsort(-contrib_at_bar)
    print(f"\n=== Bar {bar_dt_utc} (W{window}) — {len(idx)} non-reactive articles in 12h lookback ===")
    for r, k in enumerate(rank[:8], 1):
        print(f"  {r}. s={s_i[k]:5.2f}  t½={half_lives[k]:6.1f}m  age={age_min[k]:6.1f}m  contrib@bar={contrib_at_bar[k]:5.2f}  | {titles[k][:80]}")

    # ── Predictions across ±plot window
    pred = np.load(pred_path, allow_pickle=True)
    pred_dt = pd.to_datetime(pred["datetime"], utc=True)
    pred_ts = (pred_dt.astype("int64") // 10**9).values
    plot_lo = bar_ts - plot_hours_before*3600
    plot_hi = bar_ts + plot_hours_after*3600
    mask = (pred_ts >= plot_lo) & (pred_ts <= plot_hi)
    if mask.sum() == 0:
        raise RuntimeError("No test predictions in window")
    pdt = pred_dt[mask]
    ts_p = pred["ts_mlp"][mask].astype(np.float64)
    mn_p = pred["main_model"][mask].astype(np.float64)
    ac_p = pred["actual"][mask].astype(np.float64)
    ts_vol = np.sqrt(np.clip(ts_p, 1e-15, None) * PY) * 100
    mn_vol = np.sqrt(np.clip(mn_p, 1e-15, None) * PY) * 100
    ac_vol = np.sqrt(np.clip(ac_p, 1e-15, None) * PY) * 100
    bidx = int(np.argmin(np.abs(pred_ts[mask] - bar_ts)))

    # ── Plot: top forecast, bottom horizontal contribution bars
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.titleweight": "bold", "axes.titlesize": 13,
        "axes.labelsize": 12, "legend.fontsize": 11,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
    })

    n_articles_show = min(top_n_label, len(idx))
    fig_height = 4.0 + 0.45 * n_articles_show
    fig = plt.figure(figsize=(11, fig_height))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, max(1.0, 0.55*n_articles_show)],
                          hspace=0.35)

    # ── Top panel: vol forecasts ─────────────────────────────────────────────
    axT = fig.add_subplot(gs[0])
    axT.plot(pdt, ts_vol, color="#6B7280", linewidth=2.0, linestyle="--",
             label="Forecast without news")
    axT.plot(pdt, mn_vol, color="#C0392B", linewidth=2.6,
             label="Forecast with news")
    axT.plot(pdt, ac_vol, color="black", linewidth=1.4, alpha=0.55,
             label="Realized")
    axT.axvline(pd.to_datetime(bar_ts, unit="s", utc=True), color="black",
                linestyle=":", linewidth=0.9, alpha=0.45)

    # On-figure annotation at the focal bar — sharp, no monospace
    bar_x = pd.to_datetime(bar_ts, unit="s", utc=True)
    annot_text = (f"Without news: {ts_vol[bidx]:.0f}%\n"
                  f"With news:    {mn_vol[bidx]:.0f}%\n"
                  f"Realized:     {ac_vol[bidx]:.0f}%")
    axT.annotate(annot_text, xy=(bar_x, mn_vol[bidx]),
                 xytext=(20, -10), textcoords="offset points",
                 fontsize=11, family="DejaVu Sans", va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                           edgecolor="#999999", linewidth=0.8))

    axT.set_ylabel("Annualized volatility (%)")
    axT.set_title("BTC 4h vol forecast: with vs without news")
    axT.legend(loc="upper left", frameon=False)
    axT.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    axT.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axT.set_xlim(pd.to_datetime(plot_lo, unit="s", utc=True),
                 pd.to_datetime(plot_hi, unit="s", utc=True))

    # ── Bottom panel: horizontal contribution bars ──────────────────────────
    axB = fig.add_subplot(gs[1])
    show_idx = rank[:n_articles_show]
    y_pos = np.arange(len(show_idx))[::-1]  # rank 1 at top
    contribs = contrib_at_bar[show_idx]
    norm = contribs / max(contribs.max(), 1e-9)
    bar_colors = plt.cm.Reds(0.30 + 0.55 * norm)
    axB.barh(y_pos, contribs, color=bar_colors, edgecolor="#444",
             linewidth=0.6, height=0.65)

    labels = []
    for k in show_idx:
        title_short = titles[k]
        if len(title_short) > 78:
            title_short = title_short[:75] + "…"
        labels.append(title_short)
    axB.set_yticks(y_pos)
    axB.set_yticklabels(labels, fontsize=11)

    # Right-side annotations: s, t½, age
    xmax = max(contribs.max(), 1e-9)
    for j, k in enumerate(show_idx):
        y = y_pos[j]
        axB.text(contribs[j] + xmax*0.015, y,
                 f"$s$={s_i[k]:.2f}   $t_{{1/2}}$={half_lives[k]:.0f}m   age={age_min[k]:.0f}m",
                 fontsize=10, va="center", ha="left",
                 color="#444")

    axB.set_xlabel(r"Contribution to news intensity at the forecast bar: $s_i \cdot e^{-\lambda_i \, \Delta t_i}$")
    axB.set_title(f"News articles in the {LOOKBACK_H}h lookback before the forecast bar")
    axB.set_xlim(0, xmax * 1.55)
    axB.grid(axis="x", linewidth=0.5, alpha=0.6)
    axB.grid(axis="y", linewidth=0)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)

    fig.suptitle(f"BTC 4h forecast on {bar_dt_utc.strftime('%Y-%m-%d %H:%M')} UTC",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


def main():
    # Best 2025 example: Nov 3 — vol spikes & news with news correctly
    make(pd.Timestamp("2025-11-03 13:15:00", tz="UTC"), window=6,
         fig_path=OUT_DIR / "news_effect_nov2025_betterforecast.png")
    # March 5 2024: Bitcoin first record-high; huge improvement in W3
    make(pd.Timestamp("2024-03-05 18:30:00", tz="UTC"), window=3,
         fig_path=OUT_DIR / "news_effect_mar2024_recordhigh.png")


if __name__ == "__main__":
    main()
