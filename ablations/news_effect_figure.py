"""Figure: forecast WITHOUT news vs WITH news for a single high-impact event.

Picks a 4h bar that follows a major news arrival and shows:
  Top    — vol forecast over ±24h around the bar: TS-only baseline (no news)
           vs Main Model (with news), with realized vol overlaid.
  Bottom — the actual news articles in the 12h lookback, plotted as decay
           kernels s·exp(-λ·age) and labeled with their headlines.

No category colors — only the article text matters.

Outputs to results/analysis/btc_news_impact_v3/news_effect_<event>.png
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

CB_PATH   = REPO_ROOT / "data/processed/btc_news_embeddings_cryptobert.npz"
CKPT_PATH = REPO_ROOT / "results/main/btc_cryptobert_v2/checkpoints/main_model_w6_4h.pt"
PRED_PATH = REPO_ROOT / "results/main/btc_cryptobert_v2/predictions/window_6_4h_predictions.npz"
RUN_CFG   = REPO_ROOT / "results/main/btc_cryptobert_v2/run_config.json"
OUT_DIR   = REPO_ROOT / "results/analysis/btc_news_impact_v3"

LOOKBACK_H = 12
HORIZON = "4h"
PY = 365 * 24 / 4   # 4h periods/year
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
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
    with torch.no_grad():
        B = 4096
        for i in range(0, len(embeddings), B):
            x = torch.tensor(embeddings[i:i+B], dtype=torch.float32, device=DEVICE)
            out = model.news_head(model.input_norm(x))
            s[i:i+B]  = torch.nn.functional.softplus(out[:, 0]).cpu().numpy()
            lam[i:i+B] = (0.001 + 0.499 * torch.sigmoid(out[:, 1])).cpu().numpy()
    return s, lam


def make(bar_dt_utc, fig_path, plot_hours_before=12, plot_hours_after=12):
    cb = np.load(CB_PATH)
    si = np.argsort(cb["timestamps"])
    emb = cb["embeddings"][si].astype(np.float32)
    art_ts = cb["timestamps"][si].astype(np.int64)
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
    t_i  = art_ts[idx]
    s_i  = s_all[idx]
    lam_i = lam_all[idx]
    titles = titles_all[idx]
    age_min_at_bar = (bar_ts - t_i) / 60.0
    contrib_at_bar = s_i * np.exp(-lam_i * age_min_at_bar)
    rank = np.argsort(-contrib_at_bar)
    print(f"\n=== {bar_dt_utc} ({len(idx)} non-reactive articles in 12h lookback) ===")
    for r, k in enumerate(rank[:8], 1):
        hl = np.log(2) / max(lam_i[k], 1e-6)
        print(f"  {r}. s={s_i[k]:5.2f}  t½={hl:6.1f}m  age={age_min_at_bar[k]:6.1f}m  contrib@bar={contrib_at_bar[k]:5.2f}  | {titles[k][:75]}")

    # ── Load predictions over ±plot window
    pred = np.load(PRED_PATH, allow_pickle=True)
    pred_dt = pd.to_datetime(pred["datetime"], utc=True)
    pred_ts = (pred_dt.astype("int64") // 10**9).values
    plot_lo = bar_ts - plot_hours_before*3600
    plot_hi = bar_ts + plot_hours_after*3600
    mask = (pred_ts >= plot_lo) & (pred_ts <= plot_hi)
    if mask.sum() == 0:
        raise RuntimeError("No test predictions in window — pick a date inside the W6 test span.")
    pdt  = pred_dt[mask]
    ts_p = pred["ts_mlp"][mask].astype(np.float64)
    mn_p = pred["main_model"][mask].astype(np.float64)
    ac_p = pred["actual"][mask].astype(np.float64)
    ts_vol  = np.sqrt(np.clip(ts_p, 1e-15, None) * PY) * 100
    mn_vol  = np.sqrt(np.clip(mn_p, 1e-15, None) * PY) * 100
    ac_vol  = np.sqrt(np.clip(ac_p, 1e-15, None) * PY) * 100

    # Index of the focus bar in the prediction array (closest match)
    bidx = int(np.argmin(np.abs(pred_ts[mask] - bar_ts)))

    # ── Plot
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titleweight": "bold"})
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                         gridspec_kw=dict(height_ratios=[1.0, 1.4]))

    # Top: forecasts ±24h
    ax_top.plot(pdt, ts_vol, color="#6B7280", linewidth=1.8, linestyle="--",
                label="Forecast without news (TS-only)")
    ax_top.plot(pdt, mn_vol, color="#DC2626", linewidth=2.2,
                label="Forecast with news (Main Model)")
    ax_top.plot(pdt, ac_vol, color="black", linewidth=1.0, alpha=0.6,
                marker="o", markersize=3, label="Realized vol")
    ax_top.axvline(pd.to_datetime(bar_ts, unit="s", utc=True), color="black",
                   linestyle=":", linewidth=0.8, alpha=0.6)
    ax_top.set_ylabel("Annualized vol (%)")
    ax_top.set_title("Vol forecast WITHOUT news vs WITH news")
    ax_top.legend(loc="upper left", fontsize=9.5, frameon=True)
    # Annotate the focal bar's three numbers
    ax_top.annotate(
        f"At {bar_dt_utc.strftime('%H:%M')} UTC bar:\n"
        f"  without news: {ts_vol[bidx]:.1f}%\n"
        f"  with news:    {mn_vol[bidx]:.1f}%   ({(mn_vol[bidx]/ts_vol[bidx]-1)*100:+.0f}%)\n"
        f"  realized:     {ac_vol[bidx]:.1f}%",
        xy=(pd.to_datetime(bar_ts, unit="s", utc=True), mn_vol[bidx]),
        xytext=(0.62, 0.92), textcoords="axes fraction",
        fontsize=10, fontfamily="monospace", va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#cccccc", alpha=0.95),
    )

    # Bottom: per-article decay kernels + small numbered markers at arrivals.
    # Titles listed in a legend block on the right, indexed by rank.
    tau_min = np.arange(-LOOKBACK_H*60, plot_hours_after*60+1, 1)
    tau_abs = bar_ts + tau_min*60
    tau_dt  = pd.to_datetime(tau_abs, unit="s", utc=True)

    # Rank-based color (darker = bigger contributor at bar). No category info.
    norm_rank = np.zeros(len(idx))
    norm_rank[rank] = np.linspace(0.85, 0.22, len(idx))
    cmap = plt.cm.Reds
    for k in rank[::-1]:                  # plot weakest first, strongest on top
        active = tau_abs >= t_i[k]
        age_local = (tau_abs - t_i[k]) / 60.0
        kernel = s_i[k] * np.exp(-lam_i[k] * np.clip(age_local, 0, None))
        kernel[~active] = np.nan
        lw = 2.6 if k == rank[0] else (1.7 if k == rank[1] else 1.0)
        alpha = 0.98 if k == rank[0] else (0.75 if k == rank[1] else 0.55)
        ax_bot.plot(tau_dt, kernel, color=cmap(norm_rank[k]),
                    linewidth=lw, alpha=alpha,
                    zorder=10 if k == rank[0] else 5)

    # Numbered markers at each article arrival, on the x-axis (y=0).
    # Use rank index so the most important article is labeled "1".
    TOP_N_LABELS = min(5, len(idx))
    for r_pos, k in enumerate(rank[:TOP_N_LABELS], 1):
        arrive_dt = pd.to_datetime(t_i[k], unit="s", utc=True)
        ax_bot.scatter([arrive_dt], [s_i[k]], color=cmap(norm_rank[k]),
                       edgecolor="black", linewidth=0.5, s=70, zorder=11)
        ax_bot.annotate(f"{r_pos}", xy=(arrive_dt, s_i[k]),
                        xytext=(0, 6), textcoords="offset points",
                        fontsize=9, fontweight="bold",
                        ha="center", va="bottom",
                        color="black", zorder=12)

    ax_bot.axvline(pd.to_datetime(bar_ts, unit="s", utc=True),
                   color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_bot.text(pd.to_datetime(bar_ts, unit="s", utc=True),
                ax_bot.get_ylim()[1] * 0.97, " forecast bar",
                fontsize=8, ha="left", va="top", color="#555")
    ax_bot.set_ylabel(r"Per-article intensity  $s_i \cdot e^{-\lambda_i (t-t_i)}$")
    ax_bot.set_title("News articles in the 12h lookback (numbered by contribution-at-bar; full headlines below)")
    ax_bot.set_xlim(pd.to_datetime(plot_lo, unit="s", utc=True),
                    pd.to_datetime(plot_hi, unit="s", utc=True))
    ax_bot.set_xlabel(f"{bar_dt_utc.strftime('%Y-%m-%d')} (UTC)")
    ax_bot.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # Build a clean headline list below the bottom plot
    headline_lines = []
    for r_pos, k in enumerate(rank[:TOP_N_LABELS], 1):
        title_short = titles[k][:85] + ("…" if len(titles[k]) > 85 else "")
        hl = np.log(2) / max(lam_i[k], 1e-6)
        arrive_local = pd.to_datetime(t_i[k], unit="s", utc=True).strftime("%H:%M")
        headline_lines.append(
            f"{r_pos}. [{arrive_local}]  s={s_i[k]:.2f}, t½={hl:.0f}m, age@bar={age_min_at_bar[k]:.0f}m   |   {title_short}"
        )
    fig.text(0.06, -0.04, "\n".join(headline_lines),
             fontsize=9, family="monospace", va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fafafa",
                       edgecolor="#cccccc"))

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


def main():
    make(pd.Timestamp("2025-10-24 13:00:00", tz="UTC"),
         OUT_DIR / "news_effect_cpi_oct2025.png")
    make(pd.Timestamp("2025-07-03 13:00:00", tz="UTC"),
         OUT_DIR / "news_effect_nfp_jul2025.png")


if __name__ == "__main__":
    main()
