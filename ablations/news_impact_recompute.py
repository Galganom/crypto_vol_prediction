"""Recompute news impact attributions using cleaner category labels.

Loads:
  - Cleaner labels: data/processed/btc_news_categories_clean.npz
  - Trained Main Model checkpoints per window: results/main/btc_cryptobert/checkpoints/
  - Per-window PCA bases:                       results/main/btc_cryptobert/metadata/
  - CryptoBERT embeddings (raw):                data/processed/btc_news_embeddings_cryptobert.npz
  - News timestamps:                            data/processed/btc_news_embeddings_cryptobert.npz

For each (window, article) pair where the article is reachable from a test bar's
12-hr lookback, computes:
  s_i:       softplus(news_head(PCA(emb_i))[0])
  lambda_i:  0.001 + 0.499 * sigmoid(news_head(PCA(emb_i))[1])

Then per-bar attribution: for each test bar, compute decay-weighted contribution
  contrib_{i,b} = s_i * exp(-lambda_i * dt_{i,b}) * (article in lookback ind.)
The full bar log-effect is residual_head(sum_i contrib_{i,b}).clamp(-2, 2). We
allocate each article's per-bar share by  contrib_{i,b} / sum_i contrib_{i,b}.

Outputs (results/analysis/btc_news_impact_4h_v2/):
  per_article_s_lambda.csv               (window, article_id, s, lambda, half_life)
  per_article_impact.csv                 (window, article_id, sum_signed_eff, sum_abs_eff, n_bars_touched)
  category_s_lambda_stats.csv            (category, s_weighted_mean, lambda_..., half_life_...)
  window_category_s_lambda_stats.csv
  category_impact_summary.csv            (category, impact_share_abs, mean_eff_when_present, ...)
  category_distribution.csv              (category, article_count, share)
  top_event_exposures.csv                (top 250 article-bar combos)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from main_model_cryptobert import CryptoBertMainModel  # noqa: E402

# ─── Config ──────────────────────────────────────────────────────────────────
BTC_RESULTS_DIR = REPO_ROOT / "results/main/btc_cryptobert"
CHECKPOINT_DIR = BTC_RESULTS_DIR / "checkpoints"
META_DIR = BTC_RESULTS_DIR / "metadata"
PRED_DIR = BTC_RESULTS_DIR / "predictions"
HORIZON = "4h"

CRYPTOBERT_PATH = REPO_ROOT / "data/processed/btc_news_embeddings_cryptobert.npz"
LABELS_PATH = REPO_ROOT / "data/processed/btc_news_categories_clean.npz"
NEWS_PARQUET = REPO_ROOT / "data/news/processed_news.parquet"

OUT_DIR = REPO_ROOT / "results/analysis/btc_news_impact_4h_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK_SECONDS = 12 * 3600
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_run_config():
    cfg_path = BTC_RESULTS_DIR / "run_config.json"
    return json.loads(cfg_path.read_text())


def load_window_pca(w):
    p = META_DIR / f"cryptobert_pca_w{w}.npz"
    d = np.load(p)
    return {"components": d["components"], "mean": d["mean"], "k": d["components"].shape[0]}


def load_window_meta(w):
    p = META_DIR / f"window_{w}_meta.npz"
    d = np.load(p, allow_pickle=True)
    return {"train_idx": d["train_idx"], "val_idx": d["val_idx"], "test_idx": d["test_idx"]}


def load_predictions(w, horizon):
    p = PRED_DIR / f"window_{w}_{horizon}_predictions.npz"
    return dict(np.load(p, allow_pickle=True))


def transform_pca(emb, components, mean):
    """Apply per-window PCA transform: (emb - mean) @ components.T"""
    return (emb - mean) @ components.T


def main():
    cfg = load_run_config()
    n_windows = len(cfg["windows"])
    feat_dim = cfg["cryptobert_pca_k"]
    hidden = tuple(cfg["hidden"])
    print(f"Run config: {n_windows} windows, PCA k={feat_dim}, hidden={hidden}")

    # Load CryptoBERT embeddings sorted by timestamp
    cb = np.load(CRYPTOBERT_PATH)
    cb_sort = np.argsort(cb["timestamps"])
    emb_all = cb["embeddings"][cb_sort].astype(np.float32)
    news_ts_all = cb["timestamps"][cb_sort].astype(np.int64)
    n_articles = len(emb_all)
    print(f"Articles: {n_articles:,}")

    # Load cleaner labels (must be aligned to the same news ordering)
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Run relabel_news_categories.py first; missing {LABELS_PATH}")
    lab = np.load(LABELS_PATH, allow_pickle=True)
    lab_ts = lab["unix_timestamp"]
    if not np.array_equal(np.sort(lab_ts), news_ts_all):
        raise ValueError("Cleaner-label timestamps do not align with embedding timestamps")
    # Reindex labels by sorted timestamp ordering
    lab_order = np.argsort(lab_ts)
    cat_all = np.array(lab["category"][lab_order], dtype=object)
    conf_all = lab["confidence"][lab_order]

    # Load article ids/titles for the top-events output
    news_df = pd.read_parquet(NEWS_PARQUET)
    btc = news_df[news_df["asset"] == "BTC"].sort_values("unix_timestamp").reset_index(drop=True)
    titles_all = btc["title"].astype(str).values
    ids_all = btc["id"].values

    # Per-window: load checkpoint, compute s/lambda for every article
    rows_sl = []   # per-article-per-window s/lambda
    rows_imp = []  # per-article-per-window aggregated impact (sum signed / abs)
    top_events = []  # top per-bar-article exposures across all windows

    for w in cfg["windows"]:
        ckpt_path = CHECKPOINT_DIR / f"main_model_w{w}_{HORIZON}.pt"
        if not ckpt_path.exists():
            print(f"  W{w}: missing {ckpt_path}, skip"); continue
        pca = load_window_pca(w)
        if pca["k"] != feat_dim:
            print(f"  W{w}: PCA k={pca['k']} differs from cfg {feat_dim}, skip"); continue

        # Build model and load weights
        model = CryptoBertMainModel(feat_dim, hidden=hidden).to(DEVICE)
        sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(sd)
        model.eval()

        # PCA-transform the FULL corpus with this window's basis (vectorized)
        feats_w = transform_pca(emb_all, pca["components"], pca["mean"])
        feats_t = torch.tensor(feats_w, dtype=torch.float32, device=DEVICE)

        # news_head outputs (s_raw, lam_raw) per article — vectorized batches
        s_w = np.empty(n_articles, dtype=np.float32)
        lam_w = np.empty(n_articles, dtype=np.float32)
        with torch.no_grad():
            B = 4096
            for i in range(0, n_articles, B):
                out = model.news_head(feats_t[i:i + B])
                s = torch.nn.functional.softplus(out[:, 0]).cpu().numpy()
                lam = (0.001 + 0.499 * torch.sigmoid(out[:, 1])).cpu().numpy()
                s_w[i:i + B] = s
                lam_w[i:i + B] = lam

        # Predictions for this window — used to scale the signal back to log-effect
        preds = load_predictions(w, HORIZON)
        actual = preds["actual"].astype(np.float64)
        ts_pred = preds["ts_mlp"].astype(np.float64)
        main_pred = preds["main_model"].astype(np.float64)
        # Bar log effect = log(main_pred) - log(ts_pred); equals residual_head(signal).clamp(-2,2)
        bar_log_eff = np.log(np.clip(main_pred, 1e-15, None)) - np.log(np.clip(ts_pred, 1e-15, None))
        bar_log_eff = np.clip(bar_log_eff, -2.0, 2.0)

        # Test bar timestamps
        meta = load_window_meta(w)
        test_idx = meta["test_idx"]
        # The predictions npz also contains datetime + test_idx — but we need bar_ts.
        # Compute from datetime strings if present, otherwise derive.
        if "datetime" in preds:
            bar_dt = pd.to_datetime(preds["datetime"], utc=True)
            bar_ts = (bar_dt.astype("int64") // 10**9).values.astype(np.int64)
        else:
            raise RuntimeError(f"W{w}: missing datetime in predictions npz")

        # For each test bar, find articles in 12hr lookback (left-inclusive)
        left = np.searchsorted(news_ts_all, bar_ts - LOOKBACK_SECONDS, side="left")
        right = np.searchsorted(news_ts_all, bar_ts, side="left")

        # Aggregate per-article impact within this window (signed & abs)
        signed_sum = np.zeros(n_articles, dtype=np.float64)
        abs_sum = np.zeros(n_articles, dtype=np.float64)
        bars_touched = np.zeros(n_articles, dtype=np.int32)

        # Top per-bar-article exposures
        wlocal_top = []  # tuples (signal_contrib, article_log_eff, abs, signed, b_idx, art_idx)

        for b in range(len(bar_ts)):
            l, r = left[b], right[b]
            if r <= l:
                continue
            # Skip if bar's log effect is essentially zero
            if abs(bar_log_eff[b]) < 1e-9:
                continue
            arts = np.arange(l, r)
            dt_min = (bar_ts[b] - news_ts_all[arts]).astype(np.float64) / 60.0
            contrib = s_w[arts] * np.exp(-lam_w[arts] * dt_min)
            total_contrib = contrib.sum()
            if total_contrib < 1e-9:
                continue
            shares = contrib / total_contrib
            article_log_eff = shares * bar_log_eff[b]   # signed
            article_abs_eff = np.abs(article_log_eff)
            signed_sum[arts] += article_log_eff
            abs_sum[arts] += article_abs_eff
            bars_touched[arts] += 1

            # Track top exposures: pick the largest abs effect from this bar
            j = int(np.argmax(article_abs_eff))
            wlocal_top.append((arts[j], b, contrib[j], article_log_eff[j], article_abs_eff[j]))

        # Save per-article window stats
        for art_idx in np.where(bars_touched > 0)[0]:
            rows_imp.append({
                "window": int(w), "article_idx": int(art_idx),
                "id": int(ids_all[art_idx]),
                "category": cat_all[art_idx], "confidence": float(conf_all[art_idx]),
                "n_bars_touched": int(bars_touched[art_idx]),
                "sum_signed_log_eff": float(signed_sum[art_idx]),
                "sum_abs_log_eff": float(abs_sum[art_idx]),
            })

        # Per-article s/λ for this window (only for articles that appeared in any lookback)
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

        # Top per-bar-article exposures from this window
        wlocal_top.sort(key=lambda t: -t[4])
        for art_idx, b, sig, signed_eff, abs_eff in wlocal_top[:80]:
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

        print(f"  W{w}: articles in any lookback = {len(appeared):,} / {n_articles:,}")

    sl_df = pd.DataFrame(rows_sl)
    imp_df = pd.DataFrame(rows_imp)
    sl_df.to_csv(OUT_DIR / "per_article_s_lambda.csv", index=False, float_format="%.10f")
    imp_df.to_csv(OUT_DIR / "per_article_impact.csv", index=False, float_format="%.10f")
    print(f"Saved per-article tables to {OUT_DIR}")

    # ─── Aggregates ──────────────────────────────────────────────────────────
    # Distribution: count UNIQUE articles per category (using clean labels)
    dist = pd.Series(cat_all).value_counts().reset_index()
    dist.columns = ["category", "article_count"]
    dist["article_share"] = dist["article_count"] / dist["article_count"].sum()
    dist.to_csv(OUT_DIR / "category_distribution.csv", index=False, float_format="%.10f")

    # window-category s/λ stats: weighted mean by 1.0 per article
    grp = sl_df.groupby(["window", "category"])
    wcat = grp.agg(
        unique_articles=("article_idx", "nunique"),
        s_weighted_mean=("s", "mean"),
        s_weighted_std=("s", "std"),
        lambda_weighted_mean_per_min=("lambda_per_min", "mean"),
        lambda_weighted_std_per_min=("lambda_per_min", "std"),
        half_life_weighted_median_min=("half_life_min", "median"),
    ).reset_index()
    wcat["half_life_weighted_median_hours"] = wcat["half_life_weighted_median_min"] / 60.0
    wcat.to_csv(OUT_DIR / "window_category_s_lambda_stats.csv", index=False, float_format="%.10f")

    # Pooled category s/λ stats
    grp = sl_df.groupby("category")
    cat_stats = grp.agg(
        article_window_observations=("article_idx", "size"),
        unique_articles=("article_idx", "nunique"),
        s_weighted_mean=("s", "mean"),
        s_weighted_std=("s", "std"),
        lambda_weighted_mean_per_min=("lambda_per_min", "mean"),
        lambda_weighted_std_per_min=("lambda_per_min", "std"),
        half_life_weighted_median_min=("half_life_min", "median"),
    ).reset_index()
    cat_stats["half_life_weighted_median_hours"] = cat_stats["half_life_weighted_median_min"] / 60.0
    cat_stats.to_csv(OUT_DIR / "category_s_lambda_stats.csv", index=False, float_format="%.10f")

    # Pooled category impact summary
    grp = imp_df.groupby("category")
    cat_imp = grp.agg(
        unique_articles=("article_idx", "nunique"),
        bars_touched=("n_bars_touched", "sum"),
        sum_signed_log_effect=("sum_signed_log_eff", "sum"),
        sum_abs_log_effect=("sum_abs_log_eff", "sum"),
    ).reset_index()
    total_abs = cat_imp["sum_abs_log_effect"].sum()
    cat_imp["impact_share_abs"] = cat_imp["sum_abs_log_effect"] / max(total_abs, 1e-9)
    cat_imp = cat_imp.sort_values("impact_share_abs", ascending=False)
    cat_imp.to_csv(OUT_DIR / "category_impact_summary.csv", index=False, float_format="%.10f")

    # Top events
    te = pd.DataFrame(top_events).sort_values("abs_log_effect", ascending=False).head(250)
    te.to_csv(OUT_DIR / "top_event_exposures.csv", index=False, float_format="%.10f")

    print(f"\nAll outputs in {OUT_DIR}:")
    for p in sorted(OUT_DIR.glob("*.csv")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
