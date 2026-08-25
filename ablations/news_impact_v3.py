"""News impact attribution using the FULL 768-dim CryptoBERT Main Model.

Replaces news_impact_recompute.py (which targeted PCA-256). Now:
  - Loads the v2 production checkpoints from results/main/btc_cryptobert_v2/.
  - Uses raw 768-dim CryptoBERT embeddings (with the LayerNorm-input model).
  - Joins per-article s, λ, and per-bar attribution with the CLEANER tag-based
    nonreactive filter (via main_model_common.load_news_features) and the
    cleaner DeBERTa-v3-large category labels.

For each test bar in each window:
  - Find non-reactive articles in 12hr lookback
  - Compute s_i = softplus(news_head(LayerNorm(cb_emb_i))[0])
  - Compute λ_i = 0.001 + 0.499·sigmoid(...[1])
  - contrib_i = s_i · exp(-λ_i · dt_i)
  - share_i = contrib_i / Σ contrib   (signal-share allocation)
  - bar_log_eff = log(main_pred / ts_pred) clamped to [-2, 2]
  - article_log_eff_i = share_i · bar_log_eff

Outputs to results/analysis/btc_news_impact_v3/:
  per_article_s_lambda.csv          per (window, article)
  per_article_impact.csv            per (window, article) summed across bars
  category_s_lambda_stats.csv       per cleaner category
  window_category_s_lambda_stats.csv
  category_impact_summary.csv
  category_distribution.csv
  top_event_exposures.csv
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from main_model_common import (  # noqa: E402
    LOG_EPS, WINDOWS, align_bars_with_news, build_price_bars, cfg,
    load_news_features, log,
)
from main_model_cryptobert import CryptoBertMainModel  # noqa: E402

# ─── Config (defaults; can be overridden via CLI) ────────────────────────────
DEFAULT_MAIN_DIR = REPO_ROOT / "results/main/btc_cryptobert_v2"
DEFAULT_OUT_DIR = REPO_ROOT / "results/analysis/btc_news_impact_v3"
HORIZON = "4h"
LOOKBACK_SECONDS = 12 * 3600

CRYPTOBERT_PATH = REPO_ROOT / "data/processed/btc_news_embeddings_cryptobert.npz"
CLEAN_LABELS_PATH = REPO_ROOT / "data/processed/btc_news_categories_clean.npz"
NEWS_PARQUET = REPO_ROOT / "data/news/processed_news.parquet"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_window_meta(main_results_dir, w):
    p = main_results_dir / "metadata" / f"window_{w}_meta.npz"
    return dict(np.load(p, allow_pickle=True))


def load_predictions(main_results_dir, w):
    p = main_results_dir / "predictions" / f"window_{w}_{HORIZON}_predictions.npz"
    return dict(np.load(p, allow_pickle=True))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dir", default=str(DEFAULT_MAIN_DIR),
                        help="Run directory containing checkpoints/, predictions/, metadata/.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    main_results_dir = Path(args.main_dir)
    checkpoint_dir = main_results_dir / "checkpoints"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Device: {DEVICE}; horizon: {HORIZON}")
    log(f"Main dir: {main_results_dir}")
    log(f"Out dir: {out_dir}")
    cfg_path = main_results_dir / "run_config.json"
    run_cfg = json.loads(cfg_path.read_text())
    feat_dim = int(run_cfg.get("cryptobert_input_dim", 768))
    hidden = tuple(run_cfg["hidden"])
    n_windows = len(run_cfg["windows"])
    log(f"feat_dim={feat_dim}, hidden={hidden}, windows={n_windows}")

    # Load CryptoBERT embeddings (sorted by timestamp)
    cb = np.load(CRYPTOBERT_PATH)
    cb_si = np.argsort(cb["timestamps"])
    emb_all = cb["embeddings"][cb_si].astype(np.float32)
    news_ts_all = cb["timestamps"][cb_si].astype(np.int64)
    n_articles = len(emb_all)
    log(f"Articles: {n_articles:,}")

    # Cleaner labels
    if not CLEAN_LABELS_PATH.exists():
        raise FileNotFoundError(f"{CLEAN_LABELS_PATH} missing — run relabel_news_categories.py first")
    lab = np.load(CLEAN_LABELS_PATH, allow_pickle=True)
    lab_ts = lab["unix_timestamp"]
    if not np.array_equal(np.sort(lab_ts), news_ts_all):
        raise ValueError("Cleaner-label timestamps do not match embeddings")
    order = np.argsort(lab_ts)
    cat_all = np.array(lab["category"][order], dtype=object)
    conf_all = lab["confidence"][order]
    log(f"Label distribution: {dict(pd.Series(cat_all).value_counts())}")

    # Build news features (gives nonreactive_mask via tag-based filter)
    bars = build_price_bars("BTC")
    news = load_news_features("BTC")
    bars, left, right = align_bars_with_news(bars, news["timestamps"])
    nonreactive = news["nonreactive_mask"].astype(bool)
    if len(nonreactive) != n_articles:
        raise ValueError("nonreactive mask length != article count")
    log(f"Non-reactive articles (kept): {int(nonreactive.sum()):,} of {n_articles:,}")

    # Article ids/titles for top events
    nd = pd.read_parquet(NEWS_PARQUET)
    btc = nd[nd["asset"] == "BTC"].sort_values("unix_timestamp").reset_index(drop=True)
    titles_all = btc["title"].astype(str).values
    ids_all = btc["id"].values

    rows_sl = []     # per-article-per-window s/lambda
    rows_imp = []    # per-article-per-window impact
    top_events = []  # top per-bar-article exposures

    for w in run_cfg["windows"]:
        ckpt_path = checkpoint_dir / f"main_model_w{w}_{HORIZON}.pt"
        if not ckpt_path.exists():
            log(f"  W{w}: missing {ckpt_path.name}, skip"); continue
        model = CryptoBertMainModel(feat_dim, hidden=hidden).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        model.eval()

        # Per-article s, λ via news_head on LayerNormed full corpus
        s_w = np.empty(n_articles, dtype=np.float32)
        lam_w = np.empty(n_articles, dtype=np.float32)
        with torch.no_grad():
            B = 4096
            for i in range(0, n_articles, B):
                xb = torch.tensor(emb_all[i:i + B], dtype=torch.float32, device=DEVICE)
                xn = model.input_norm(xb)
                out = model.news_head(xn)
                s = torch.nn.functional.softplus(out[:, 0]).cpu().numpy()
                lam = (0.001 + 0.499 * torch.sigmoid(out[:, 1])).cpu().numpy()
                s_w[i:i + B] = s
                lam_w[i:i + B] = lam

        # Predictions
        preds = load_predictions(main_results_dir, w)
        actual = preds["actual"].astype(np.float64)
        ts_pred = preds["ts_mlp"].astype(np.float64)
        main_pred = preds["main_model"].astype(np.float64)
        bar_log_eff = np.log(np.clip(main_pred, 1e-15, None)) - np.log(np.clip(ts_pred, 1e-15, None))
        bar_log_eff = np.clip(bar_log_eff, -2.0, 2.0)
        bar_dt = pd.to_datetime(preds["datetime"], utc=True)
        bar_ts = (bar_dt.astype("int64") // 10**9).values.astype(np.int64)

        left_idx = np.searchsorted(news_ts_all, bar_ts - LOOKBACK_SECONDS, side="left")
        right_idx = np.searchsorted(news_ts_all, bar_ts, side="left")

        signed_sum = np.zeros(n_articles, dtype=np.float64)
        abs_sum = np.zeros(n_articles, dtype=np.float64)
        bars_touched = np.zeros(n_articles, dtype=np.int32)
        wlocal_top = []

        for b in range(len(bar_ts)):
            l, r = left_idx[b], right_idx[b]
            if r <= l: continue
            if abs(bar_log_eff[b]) < 1e-9: continue
            arts = np.arange(l, r)
            arts = arts[nonreactive[arts]]
            if len(arts) == 0: continue
            dt_min = (bar_ts[b] - news_ts_all[arts]).astype(np.float64) / 60.0
            contrib = s_w[arts] * np.exp(-lam_w[arts] * dt_min)
            total = contrib.sum()
            if total < 1e-9: continue
            shares = contrib / total
            art_log_eff = shares * bar_log_eff[b]
            art_abs_eff = np.abs(art_log_eff)
            signed_sum[arts] += art_log_eff
            abs_sum[arts] += art_abs_eff
            bars_touched[arts] += 1
            j = int(np.argmax(art_abs_eff))
            wlocal_top.append((arts[j], b, contrib[j], art_log_eff[j], art_abs_eff[j]))

        for art_idx in np.where(bars_touched > 0)[0]:
            rows_imp.append({
                "window": int(w), "article_idx": int(art_idx),
                "id": int(ids_all[art_idx]),
                "category": cat_all[art_idx], "confidence": float(conf_all[art_idx]),
                "n_bars_touched": int(bars_touched[art_idx]),
                "sum_signed_log_eff": float(signed_sum[art_idx]),
                "sum_abs_log_eff": float(abs_sum[art_idx]),
            })
        appeared = np.where(bars_touched > 0)[0]
        for art_idx in appeared:
            rows_sl.append({
                "window": int(w), "article_idx": int(art_idx),
                "id": int(ids_all[art_idx]),
                "category": cat_all[art_idx], "confidence": float(conf_all[art_idx]),
                "title": titles_all[art_idx],
                "s": float(s_w[art_idx]),
                "lambda_per_min": float(lam_w[art_idx]),
                "half_life_min": float(np.log(2) / lam_w[art_idx]),
            })
        wlocal_top.sort(key=lambda t: -t[4])
        for art_idx, b, sig, signed_eff, abs_eff in wlocal_top[:100]:
            top_events.append({
                "window": int(w),
                "bar_datetime": str(pd.Timestamp(bar_ts[b], unit="s", tz="UTC")),
                "article_unix_timestamp": int(news_ts_all[art_idx]),
                "category": cat_all[art_idx], "confidence": float(conf_all[art_idx]),
                "title": titles_all[art_idx],
                "age_minutes": float((bar_ts[b] - news_ts_all[art_idx]) / 60.0),
                "article_signal": float(sig),
                "signed_log_effect": float(signed_eff),
                "abs_log_effect": float(abs_eff),
            })

        log(f"  W{w}: articles in any lookback = {len(appeared):,} / {n_articles:,}")

    sl_df = pd.DataFrame(rows_sl)
    imp_df = pd.DataFrame(rows_imp)
    sl_df.to_csv(out_dir / "per_article_s_lambda.csv", index=False, float_format="%.10f")
    imp_df.to_csv(out_dir / "per_article_impact.csv", index=False, float_format="%.10f")

    # Aggregates
    dist = pd.Series(cat_all).value_counts().reset_index()
    dist.columns = ["category", "article_count"]
    dist["article_share"] = dist["article_count"] / dist["article_count"].sum()
    dist.to_csv(out_dir / "category_distribution.csv", index=False, float_format="%.10f")

    wcat = (sl_df.groupby(["window", "category"])
                  .agg(unique_articles=("article_idx", "nunique"),
                       s_weighted_mean=("s", "mean"),
                       s_weighted_std=("s", "std"),
                       lambda_weighted_mean_per_min=("lambda_per_min", "mean"),
                       lambda_weighted_std_per_min=("lambda_per_min", "std"),
                       half_life_weighted_median_min=("half_life_min", "median"))
                  .reset_index())
    wcat["half_life_weighted_median_hours"] = wcat["half_life_weighted_median_min"] / 60.0
    wcat.to_csv(out_dir / "window_category_s_lambda_stats.csv", index=False, float_format="%.10f")

    cat_stats = (sl_df.groupby("category")
                      .agg(article_window_observations=("article_idx", "size"),
                           unique_articles=("article_idx", "nunique"),
                           s_weighted_mean=("s", "mean"),
                           s_weighted_std=("s", "std"),
                           lambda_weighted_mean_per_min=("lambda_per_min", "mean"),
                           lambda_weighted_std_per_min=("lambda_per_min", "std"),
                           half_life_weighted_median_min=("half_life_min", "median"))
                      .reset_index())
    cat_stats["half_life_weighted_median_hours"] = cat_stats["half_life_weighted_median_min"] / 60.0
    cat_stats.to_csv(out_dir / "category_s_lambda_stats.csv", index=False, float_format="%.10f")

    cat_imp = (imp_df.groupby("category")
                     .agg(unique_articles=("article_idx", "nunique"),
                          bars_touched=("n_bars_touched", "sum"),
                          sum_signed_log_effect=("sum_signed_log_eff", "sum"),
                          sum_abs_log_effect=("sum_abs_log_eff", "sum"))
                     .reset_index())
    total_abs = cat_imp["sum_abs_log_effect"].sum()
    cat_imp["impact_share_abs"] = cat_imp["sum_abs_log_effect"] / max(total_abs, 1e-9)
    cat_imp = cat_imp.sort_values("impact_share_abs", ascending=False)
    cat_imp.to_csv(out_dir / "category_impact_summary.csv", index=False, float_format="%.10f")

    te = pd.DataFrame(top_events).sort_values("abs_log_effect", ascending=False).head(250)
    te.to_csv(out_dir / "top_event_exposures.csv", index=False, float_format="%.10f")

    log(f"\nAll outputs in {out_dir}")
    for p in sorted(out_dir.glob("*.csv")):
        log(f"  {p.name}")


if __name__ == "__main__":
    main()
