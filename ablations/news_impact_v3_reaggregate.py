"""Re-aggregate the existing per-article s/λ and impact tables by NEW (12-way)
category labels — no inference needed.

Reads:
  results/analysis/btc_news_impact_v3/per_article_s_lambda.csv
  results/analysis/btc_news_impact_v3/per_article_impact.csv
  data/processed/btc_news_categories_clean.npz   (must already be re-labeled)

Writes (overwrites):
  results/analysis/btc_news_impact_v3/category_s_lambda_stats.csv
  results/analysis/btc_news_impact_v3/window_category_s_lambda_stats.csv
  results/analysis/btc_news_impact_v3/category_impact_summary.csv
  results/analysis/btc_news_impact_v3/category_distribution.csv

Also re-writes per_article_s_lambda.csv / per_article_impact.csv with the
new category column replacing the old one (in-place).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results/analysis/btc_news_impact_v3"
LAB_PATH = REPO_ROOT / "data/processed/btc_news_categories_clean.npz"
NEWS_PATH = REPO_ROOT / "data/news/processed_news.parquet"


def main():
    sl = pd.read_csv(OUT_DIR / "per_article_s_lambda.csv")
    imp = pd.read_csv(OUT_DIR / "per_article_impact.csv")
    print(f"sl rows: {len(sl):,}   imp rows: {len(imp):,}")
    print(f"sl columns: {list(sl.columns)}")

    # New labels (sorted by timestamp, matching the embedding corpus order)
    lab = np.load(LAB_PATH, allow_pickle=True)
    ts = lab["unix_timestamp"]
    order = np.argsort(ts)
    cats = np.array(lab["category"][order], dtype=object)
    confs = lab["confidence"][order]
    print(f"labels: {len(cats):,}; distribution:")
    print(pd.Series(cats).value_counts().to_string())

    # Map article_idx → new (category, confidence) (article_idx is corpus order)
    cat_by_idx = pd.Series(cats, index=np.arange(len(cats)))
    conf_by_idx = pd.Series(confs, index=np.arange(len(confs)))

    sl["category"] = sl["article_idx"].map(cat_by_idx)
    sl["confidence"] = sl["article_idx"].map(conf_by_idx)
    imp["category"] = imp["article_idx"].map(cat_by_idx)
    imp["confidence"] = imp["article_idx"].map(conf_by_idx)

    # Save the re-stamped per-article tables
    sl.to_csv(OUT_DIR / "per_article_s_lambda.csv", index=False, float_format="%.10f")
    imp.to_csv(OUT_DIR / "per_article_impact.csv", index=False, float_format="%.10f")

    # Category distribution (whole corpus)
    dist = pd.Series(cats).value_counts().reset_index()
    dist.columns = ["category", "article_count"]
    dist["article_share"] = dist["article_count"] / dist["article_count"].sum()
    dist.to_csv(OUT_DIR / "category_distribution.csv", index=False, float_format="%.10f")

    # Per-window per-category s/λ stats
    wcat = (sl.groupby(["window", "category"])
              .agg(unique_articles=("article_idx", "nunique"),
                   s_weighted_mean=("s", "mean"),
                   s_weighted_std=("s", "std"),
                   lambda_weighted_mean_per_min=("lambda_per_min", "mean"),
                   lambda_weighted_std_per_min=("lambda_per_min", "std"),
                   half_life_weighted_median_min=("half_life_min", "median"))
              .reset_index())
    wcat["half_life_weighted_median_hours"] = wcat["half_life_weighted_median_min"] / 60.0
    wcat.to_csv(OUT_DIR / "window_category_s_lambda_stats.csv", index=False, float_format="%.10f")

    # Pooled per-category s/λ stats
    cat_stats = (sl.groupby("category")
                   .agg(article_window_observations=("article_idx", "size"),
                        unique_articles=("article_idx", "nunique"),
                        s_weighted_mean=("s", "mean"),
                        s_weighted_std=("s", "std"),
                        lambda_weighted_mean_per_min=("lambda_per_min", "mean"),
                        lambda_weighted_std_per_min=("lambda_per_min", "std"),
                        half_life_weighted_median_min=("half_life_min", "median"))
                   .reset_index())
    cat_stats["half_life_weighted_median_hours"] = cat_stats["half_life_weighted_median_min"] / 60.0
    cat_stats.to_csv(OUT_DIR / "category_s_lambda_stats.csv", index=False, float_format="%.10f")

    # Category impact summary
    cat_imp = (imp.groupby("category")
                  .agg(unique_articles=("article_idx", "nunique"),
                       bars_touched=("n_bars_touched", "sum"),
                       sum_signed_log_effect=("sum_signed_log_eff", "sum"),
                       sum_abs_log_effect=("sum_abs_log_eff", "sum"))
                  .reset_index()
                  .sort_values("sum_abs_log_effect", ascending=False))
    cat_imp["impact_share_abs"] = (cat_imp["sum_abs_log_effect"] /
                                    cat_imp["sum_abs_log_effect"].sum())
    cat_imp.to_csv(OUT_DIR / "category_impact_summary.csv", index=False, float_format="%.10f")

    print("\n=== category_s_lambda_stats.csv ===")
    print(cat_stats.sort_values("s_weighted_mean", ascending=False).to_string(index=False))
    print("\n=== category_impact_summary.csv ===")
    print(cat_imp.to_string(index=False))


if __name__ == "__main__":
    main()
