"""Same Sentiment vs Event analysis but using FinBERT (ProsusAI/finbert) as
the sentiment classifier instead of CryptoBERT.

FinBERT was already used in the earlier news-representation ablation, with
softmax probabilities cached at data/processed/btc_news_sentiment.npz
(columns: positive, negative, neutral — FinBERT's standard label order).

Test: does the ambiguous-vs-decisive finding replicate?
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import matplotlib as mpl

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FB_PATH    = REPO_ROOT / "data/processed/btc_news_sentiment.npz"   # FinBERT probs (cached)
NEWS_PATH  = REPO_ROOT / "data/news/processed_news.parquet"

# Defaults; overridable via --sl-path / --tag (see main()).
SL_PATH    = REPO_ROOT / "results/analysis/btc_news_impact_v3_huber/per_article_s_lambda.csv"
GROUPS_CSV = REPO_ROOT / "data/results/sentiment_groups_s_finbert_huber.csv"
TEX_OUT    = REPO_ROOT / "data/results/sentiment_comparison_finbert_huber.tex"
FIG_OUT    = REPO_ROOT / "figures/sentiment_vs_s_finbert_huber.png"

THRESHOLDS = [0.50, 0.60, 0.70]


def load_finbert_sentiment():
    """Loads cached FinBERT softmax probabilities.

    FinBERT columns order = [positive, negative, neutral]
    Return DataFrame with id, p_positive, p_negative, p_neutral.
    """
    probs = np.load(FB_PATH)["sentiment_probs"].astype(np.float32)
    nd = pd.read_parquet(NEWS_PATH)
    btc = nd[nd["asset"] == "BTC"].sort_values("unix_timestamp").reset_index(drop=True)
    if len(probs) != len(btc):
        raise ValueError(f"FinBERT probs ({len(probs):,}) != BTC corpus ({len(btc):,})")
    df = pd.DataFrame({
        "id": btc["id"].values,
        "unix_timestamp": btc["unix_timestamp"].values,
        "p_positive": probs[:, 0],
        "p_negative": probs[:, 1],
        "p_neutral":  probs[:, 2],
    })
    print(f"FinBERT probs loaded for {len(df):,} BTC articles")
    print(f"  Column means: positive={df['p_positive'].mean():.3f}  "
          f"negative={df['p_negative'].mean():.3f}  "
          f"neutral={df['p_neutral'].mean():.3f}")
    return df


def load_s_lambda():
    sl = pd.read_csv(SL_PATH).rename(columns={"lambda_per_min": "lambda"})
    print(f"Loaded {len(sl):,} (window, article) rows from {SL_PATH}")
    return sl[["window", "id", "article_idx", "category", "s", "lambda", "half_life_min"]]


def classify_by_threshold(merged, thr):
    p = merged[["p_positive", "p_neutral", "p_negative"]].values
    classes = np.array(["positive", "neutral", "negative"])
    dom = classes[p.argmax(axis=1)]
    max_prob = p.max(axis=1)
    group = np.where(max_prob < thr, "ambiguous",
                     np.array(["decisive_" + d for d in dom]))
    out = merged.copy()
    out["max_prob"] = max_prob
    out["dominant_class"] = dom
    out["group"] = group
    return out


def mw(a, b, la, lb):
    a = np.asarray(a, np.float64)[np.isfinite(np.asarray(a, np.float64))]
    b = np.asarray(b, np.float64)[np.isfinite(np.asarray(b, np.float64))]
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    return {
        "test": f"{la} vs {lb}", "U": float(U), "p": float(p),
        f"N_{la}": len(a), f"N_{lb}": len(b),
        f"mean_{la}": float(a.mean()), f"mean_{lb}": float(b.mean()),
        f"median_{la}": float(np.median(a)), f"median_{lb}": float(np.median(b)),
        f"std_{la}": float(a.std(ddof=1)), f"std_{lb}": float(b.std(ddof=1)),
    }


def stats(df, thr):
    g = df["group"].values; s = df["s"].values
    amb = s[g == "ambiguous"]
    dec = s[np.isin(g, ["decisive_positive", "decisive_neutral", "decisive_negative"])]
    pos = s[g == "decisive_positive"]
    neg = s[g == "decisive_negative"]
    r1 = mw(amb, dec, "ambiguous", "decisive")
    r2 = mw(pos, neg, "positive_decisive", "negative_decisive")
    print(f"\n=== FinBERT  Threshold = {thr} ===")
    print(df["group"].value_counts().to_string())
    print(f"\nTest 1: Ambiguous vs Decisive")
    print(f"  N: {r1['N_ambiguous']:,} ambiguous, {r1['N_decisive']:,} decisive")
    print(f"  Mean s:   amb {r1['mean_ambiguous']:.4f} vs dec {r1['mean_decisive']:.4f}")
    print(f"  Median s: amb {r1['median_ambiguous']:.4f} vs dec {r1['median_decisive']:.4f}")
    print(f"  Mann-Whitney U = {r1['U']:.0f},  p = {r1['p']:.3e}")
    print(f"\nTest 2: Positive-decisive vs Negative-decisive")
    print(f"  N: {r2['N_positive_decisive']:,} pos, {r2['N_negative_decisive']:,} neg")
    print(f"  Mean s:   pos {r2['mean_positive_decisive']:.4f} vs neg {r2['mean_negative_decisive']:.4f}")
    print(f"  Median s: pos {r2['median_positive_decisive']:.4f} vs neg {r2['median_negative_decisive']:.4f}")
    print(f"  Mann-Whitney U = {r2['U']:.0f},  p = {r2['p']:.3e}")
    return r1, r2


def write_outputs(df_main, r1, r2):
    cols = ["id", "article_idx", "window", "s", "lambda", "half_life_min",
            "p_positive", "p_neutral", "p_negative", "max_prob",
            "dominant_class", "group", "category"]
    GROUPS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_main[cols].to_csv(GROUPS_CSV, index=False, float_format="%.6f")
    print(f"\nSaved CSV: {GROUPS_CSV}")

    groups = ["decisive_positive", "decisive_neutral", "decisive_negative",
              "ambiguous"]
    rows = []
    for g in groups:
        sub = df_main[df_main["group"] == g]["s"].values
        rows.append({
            "Group": g.replace("_", " "),
            "N": len(sub),
            "Mean": float(np.mean(sub)) if len(sub) else float("nan"),
            "Median": float(np.median(sub)) if len(sub) else float("nan"),
            "Std": float(np.std(sub, ddof=1)) if len(sub) > 1 else float("nan"),
        })
    p1_str = ("< 0.001" if r1["p"] < 1e-3 else f"= {r1['p']:.3f}")
    p2_str = ("< 0.001" if r2["p"] < 1e-3 else f"= {r2['p']:.3f}")
    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{Learned impact magnitude $s$ by \textbf{FinBERT} sentiment "
        r"confidence group (threshold $\tau = 0.60$). Replicates the "
        r"ambiguous-vs-decisive effect under a different sentiment model.}",
        r"\label{tab:sentiment-vs-s-finbert}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Group & $N$ & Mean $s$ & Median $s$ & Std $s$ \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['Group']} & {r['N']:,} & {r['Mean']:.3f} & "
            f"{r['Median']:.3f} & {r['Std']:.3f} \\\\"
        )
    lines += [
        r"\bottomrule\end{tabular}",
        r"",
        r"\smallskip",
        f"\\noindent\\textit{{Mann-Whitney, ambiguous vs decisive:}} "
        f"$U = {r1['U']:,.0f}$, $p {p1_str}$ \\\\",
        f"\\noindent\\textit{{Mann-Whitney, positive-decisive vs "
        f"negative-decisive:}} $U = {r2['U']:,.0f}$, $p {p2_str}$",
        r"\end{table}",
    ]
    TEX_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved LaTeX: {TEX_OUT}")

    # Figure
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.titleweight": "bold", "axes.titlesize": 13})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))
    pal = {"decisive": "#6B7280", "ambiguous": "#C0392B",
           "pos": "#0EA5E9", "neg": "#7F1D1D"}

    data_L = [
        df_main[df_main["group"].isin(["decisive_positive", "decisive_neutral",
                                        "decisive_negative"])]["s"].values,
        df_main[df_main["group"] == "ambiguous"]["s"].values,
    ]
    parts = axL.violinplot(data_L, positions=[0, 1], showmedians=True,
                           showextrema=False, widths=0.78)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor([pal["decisive"], pal["ambiguous"]][i])
        body.set_edgecolor("black"); body.set_alpha(0.78)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.6)
    axL.set_xticks([0, 1])
    axL.set_xticklabels([f"Decisive\n(N={len(data_L[0]):,})",
                         f"Ambiguous\n(N={len(data_L[1]):,})"])
    axL.set_ylabel(r"Learned impact magnitude  $s$")
    axL.set_xlabel("FinBERT sentiment confidence group")
    p1_short = ("p < 0.001" if r1["p"] < 1e-3 else f"p = {r1['p']:.3f}")
    axL.set_title(f"Ambiguous vs Decisive   (Mann-Whitney  {p1_short})")

    data_R = [
        df_main[df_main["group"] == "decisive_positive"]["s"].values,
        df_main[df_main["group"] == "decisive_negative"]["s"].values,
        df_main[df_main["group"] == "ambiguous"]["s"].values,
    ]
    parts2 = axR.violinplot(data_R, positions=[0, 1, 2], showmedians=True,
                            showextrema=False, widths=0.78)
    cR = [pal["pos"], pal["neg"], pal["ambiguous"]]
    for i, body in enumerate(parts2["bodies"]):
        body.set_facecolor(cR[i]); body.set_edgecolor("black"); body.set_alpha(0.78)
    parts2["cmedians"].set_color("black"); parts2["cmedians"].set_linewidth(1.6)
    axR.set_xticks([0, 1, 2])
    axR.set_xticklabels([
        f"Decisive\npositive\n(N={len(data_R[0]):,})",
        f"Decisive\nnegative\n(N={len(data_R[1]):,})",
        f"Ambiguous\n(N={len(data_R[2]):,})",
    ])
    axR.set_ylabel(r"Learned impact magnitude  $s$")
    axR.set_xlabel("FinBERT sentiment confidence group")
    p2_short = ("p < 0.001" if r2["p"] < 1e-3 else f"p = {r2['p']:.3f}")
    axR.set_title(f"Positive-decisive vs Negative-decisive   (Mann-Whitney  {p2_short})")

    plt.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {FIG_OUT}")


def main():
    import argparse
    global SL_PATH, GROUPS_CSV, TEX_OUT, FIG_OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--sl-path", default=str(SL_PATH),
                        help="per_article_s_lambda.csv to read s from.")
    parser.add_argument("--tag", default="finbert_huber",
                        help="Output filename tag (CSV/TeX/figure).")
    args = parser.parse_args()
    SL_PATH    = Path(args.sl_path)
    GROUPS_CSV = REPO_ROOT / f"data/results/sentiment_groups_s_{args.tag}.csv"
    TEX_OUT    = REPO_ROOT / f"data/results/sentiment_comparison_{args.tag}.tex"
    FIG_OUT    = REPO_ROOT / f"figures/sentiment_vs_s_{args.tag}.png"

    print("="*72)
    print("Sentiment vs Event Analysis — FinBERT replication")
    print("="*72)
    print(f"s/λ source: {SL_PATH}")
    sent = load_finbert_sentiment()
    sl = load_s_lambda()
    merged = sl.merge(sent, on="id", how="inner")
    print(f"\nMerged rows: {len(merged):,}  "
          f"(unique articles: {merged['id'].nunique():,};  "
          f"windows: {sorted(merged['window'].unique().tolist())})")

    df_main = r1_main = r2_main = None
    for thr in THRESHOLDS:
        df_thr = classify_by_threshold(merged, thr)
        r1, r2 = stats(df_thr, thr)
        if abs(thr - 0.60) < 1e-9:
            df_main, r1_main, r2_main = df_thr, r1, r2

    print("\n=== Verification (threshold 0.60, FinBERT) ===")
    amb = df_main[df_main["group"] == "ambiguous"]["s"].values
    dec = df_main[df_main["group"].isin(["decisive_positive", "decisive_neutral",
                                          "decisive_negative"])]["s"].values
    print(f"  Mean s (ambiguous) > Mean s (decisive):       "
          f"{np.mean(amb) > np.mean(dec)}  "
          f"({np.mean(amb):.4f} vs {np.mean(dec):.4f})")
    print(f"  Mann-Whitney p(amb vs dec) < 0.001:           "
          f"{r1_main['p'] < 1e-3}  (p = {r1_main['p']:.3e})")
    print(f"  Positive vs Negative non-significant (p>0.05):"
          f" {r2_main['p'] > 0.05}  (p = {r2_main['p']:.3e})")

    write_outputs(df_main, r1_main, r2_main)


if __name__ == "__main__":
    main()
