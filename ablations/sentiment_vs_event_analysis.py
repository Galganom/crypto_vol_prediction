"""Sentiment vs Event analysis for the NED paper.

Tests whether the learned per-article impact magnitude `s` is driven by
CryptoBERT sentiment confidence or by sentiment ambiguity.
Hypothesis: articles where CryptoBERT cannot decisively assign
bullish/neutral/bearish receive larger learned `s` than decisive articles.

Tasks:
  1. Extract CryptoBERT 3-class sentiment probabilities for every BTC article.
  2. Reuse the per-window (s, λ) already computed by news_impact_v3.
  3. Classify articles by max_prob threshold → groups.
  4. Mann-Whitney U test:  ambiguous vs decisive(pooled)
                            bullish-decisive vs bearish-decisive
  5. Robustness across thresholds 0.5, 0.6, 0.7.
  6. Outputs:
       data/results/sentiment_groups_s.csv
       data/results/sentiment_comparison.tex
       figures/sentiment_vs_s.png

Paths (BTC 4h NED v2 run):
  - results/main/btc_cryptobert_v2/checkpoints/main_model_w{1..6}_4h.pt
  - data/processed/btc_news_embeddings_cryptobert.npz
  - data/news/processed_news.parquet
  - results/analysis/btc_news_impact_v3/per_article_s_lambda.csv  (Task-2 output)
"""
import sys, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import matplotlib as mpl

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

NEWS_PATH = REPO_ROOT / "data/news/processed_news.parquet"
SENT_OUT  = REPO_ROOT / "data/results/btc_sentiment_scores.parquet"
SL_PATH   = REPO_ROOT / "results/analysis/btc_news_impact_v3/per_article_s_lambda.csv"
GROUPS_CSV = REPO_ROOT / "data/results/sentiment_groups_s.csv"
TEX_OUT    = REPO_ROOT / "data/results/sentiment_comparison.tex"
FIG_OUT    = REPO_ROOT / "figures/sentiment_vs_s.png"

MODEL_NAME = "ElKulako/cryptobert"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
MAX_LEN = 512

THRESHOLDS = [0.50, 0.60, 0.70]


# ─── Task 1: CryptoBERT sentiment inference ────────────────────────────────
def extract_sentiment_scores():
    if SENT_OUT.exists():
        print(f"Sentiment scores already exist at {SENT_OUT}; skipping inference.")
        return pd.read_parquet(SENT_OUT)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"Loading CryptoBERT classification head from {MODEL_NAME} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    id2label = model.config.id2label
    print(f"  id2label = {id2label}")

    # Determine the index of each class
    label_to_idx = {v.lower(): k for k, v in id2label.items()}
    print(f"  label_to_idx = {label_to_idx}")
    def _pick(*names):
        for n in names:
            if n in label_to_idx:
                return label_to_idx[n]
        return None
    bullish_idx = _pick("bullish", "positive")
    neutral_idx = _pick("neutral")
    bearish_idx = _pick("bearish", "negative")
    assert all(x is not None for x in [bullish_idx, neutral_idx, bearish_idx]), \
        f"Could not resolve all 3 classes from {id2label}"
    print(f"  bullish={bullish_idx}, neutral={neutral_idx}, bearish={bearish_idx}")

    df = pd.read_parquet(NEWS_PATH)
    df = df[df["asset"] == "BTC"].sort_values("unix_timestamp").reset_index(drop=True)
    df["text_for_clf"] = (df["title"].fillna("").astype(str) + ". " +
                          df["cleaned_text_truncated"].fillna("").astype(str))
    n = len(df)
    print(f"  scoring {n:,} BTC articles ...")

    out = {"id": df["id"].values, "unix_timestamp": df["unix_timestamp"].values,
           "p_bullish": np.zeros(n, dtype=np.float32),
           "p_neutral": np.zeros(n, dtype=np.float32),
           "p_bearish": np.zeros(n, dtype=np.float32)}

    texts = df["text_for_clf"].tolist()
    softmax = torch.nn.Softmax(dim=-1)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_LEN).to(DEVICE)
            logits = model(**enc).logits
            probs = softmax(logits).cpu().numpy()
            out["p_bullish"][i:i+len(batch)] = probs[:, bullish_idx]
            out["p_neutral"][i:i+len(batch)] = probs[:, neutral_idx]
            out["p_bearish"][i:i+len(batch)] = probs[:, bearish_idx]
            if (i // BATCH_SIZE) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + len(batch)) / max(elapsed, 1e-6)
                eta = (n - (i + len(batch))) / max(rate, 1e-6)
                print(f"  {i+len(batch):,}/{n:,}  ({rate:.1f} it/s, ETA {eta/60:.1f} min)")
    sent = pd.DataFrame(out)
    SENT_OUT.parent.mkdir(parents=True, exist_ok=True)
    sent.to_parquet(SENT_OUT)
    print(f"Saved: {SENT_OUT}")
    return sent


# ─── Task 2: per-window (s, λ) — reuse existing news_impact_v3 output ─────
def load_s_lambda_per_window():
    sl = pd.read_csv(SL_PATH)
    print(f"\nLoaded {len(sl):,} (window, article) rows from {SL_PATH}")
    # Standardize columns:  id, window, s, lambda
    sl = sl.rename(columns={"lambda_per_min": "lambda"})
    return sl[["window", "id", "article_idx", "category", "s", "lambda", "half_life_min"]]


# ─── Task 3-5: classify and test ──────────────────────────────────────────
def classify_by_threshold(merged, thr):
    p = merged[["p_bullish", "p_neutral", "p_bearish"]].values
    max_prob = p.max(axis=1)
    dom = np.array(["bullish", "neutral", "bearish"])[p.argmax(axis=1)]
    group = np.where(max_prob < thr, "ambiguous",
                     np.array(["decisive_" + d for d in dom]))
    out = merged.copy()
    out["max_prob"] = max_prob
    out["dominant_class"] = dom
    out["group"] = group
    return out


def mw_test(a, b, label_a, label_b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    return {
        "test": f"{label_a} vs {label_b}",
        "U": float(U), "p": float(p),
        f"N_{label_a}": int(len(a)), f"N_{label_b}": int(len(b)),
        f"mean_{label_a}": float(a.mean()), f"mean_{label_b}": float(b.mean()),
        f"median_{label_a}": float(np.median(a)), f"median_{label_b}": float(np.median(b)),
        f"std_{label_a}": float(a.std(ddof=1)), f"std_{label_b}": float(b.std(ddof=1)),
    }


def run_stats_block(df, thr):
    g = df["group"].values
    s = df["s"].values
    amb = s[g == "ambiguous"]
    dec = s[np.isin(g, ["decisive_bullish", "decisive_neutral", "decisive_bearish"])]
    bull = s[g == "decisive_bullish"]
    bear = s[g == "decisive_bearish"]

    r1 = mw_test(amb, dec, "ambiguous", "decisive")
    r2 = mw_test(bull, bear, "bullish_decisive", "bearish_decisive")

    print(f"\n=== Threshold = {thr} ===")
    print(f"Group counts:")
    print(df["group"].value_counts().to_string())
    print(f"\nTest 1: Ambiguous vs Decisive")
    print(f"  N: {r1['N_ambiguous']:,} ambiguous, {r1['N_decisive']:,} decisive")
    print(f"  Mean s: ambiguous {r1['mean_ambiguous']:.4f} vs decisive {r1['mean_decisive']:.4f}")
    print(f"  Median s: ambiguous {r1['median_ambiguous']:.4f} vs decisive {r1['median_decisive']:.4f}")
    print(f"  Mann-Whitney U = {r1['U']:.0f},  p = {r1['p']:.3e}")
    print(f"\nTest 2: Bullish-decisive vs Bearish-decisive")
    print(f"  N: {r2['N_bullish_decisive']:,} bull, {r2['N_bearish_decisive']:,} bear")
    print(f"  Mean s: bull {r2['mean_bullish_decisive']:.4f} vs bear {r2['mean_bearish_decisive']:.4f}")
    print(f"  Median s: bull {r2['median_bullish_decisive']:.4f} vs bear {r2['median_bearish_decisive']:.4f}")
    print(f"  Mann-Whitney U = {r2['U']:.0f},  p = {r2['p']:.3e}")

    return r1, r2


# ─── Task 6: outputs (CSV, TEX, figure) ───────────────────────────────────
def write_outputs(df_main, r1, r2):
    cols = ["id", "article_idx", "window", "s", "lambda", "half_life_min",
            "p_bullish", "p_neutral", "p_bearish", "max_prob",
            "dominant_class", "group", "category"]
    GROUPS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_main[cols].to_csv(GROUPS_CSV, index=False, float_format="%.6f")
    print(f"\nSaved CSV: {GROUPS_CSV}")

    # LaTeX table
    groups = ["decisive_bullish", "decisive_neutral", "decisive_bearish",
              "ambiguous"]
    rows = []
    for g in groups:
        sub = df_main[df_main["group"] == g]["s"].values
        rows.append({
            "Group": g.replace("_", " "),
            "N": len(sub),
            "Mean s": float(np.mean(sub)) if len(sub) else float("nan"),
            "Median s": float(np.median(sub)) if len(sub) else float("nan"),
            "Std s": float(np.std(sub, ddof=1)) if len(sub) > 1 else float("nan"),
        })
    tab = pd.DataFrame(rows)
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Learned impact magnitude $s$ by CryptoBERT sentiment "
        r"confidence group (threshold $\tau = 0.60$).  Ambiguous articles "
        r"(max softmax $< 0.60$) carry significantly larger $s$ than decisive "
        r"articles in any single direction.}",
        r"\label{tab:sentiment-vs-s}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Group & $N$ & Mean $s$ & Median $s$ & Std $s$ \\",
        r"\midrule",
    ]
    for r in rows:
        tex_lines.append(
            f"{r['Group']} & {r['N']:,} & {r['Mean s']:.3f} & "
            f"{r['Median s']:.3f} & {r['Std s']:.3f} \\\\"
        )
    p1_str = ("< 0.001" if r1["p"] < 1e-3 else f"= {r1['p']:.3f}")
    p2_str = ("< 0.001" if r2["p"] < 1e-3 else f"= {r2['p']:.3f}")
    tex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\smallskip",
        f"\\noindent\\textit{{Mann-Whitney test, ambiguous vs decisive:}} "
        f"$U = {r1['U']:,.0f}$, $p {p1_str}$ \\\\",
        f"\\noindent\\textit{{Mann-Whitney test, bullish-decisive vs "
        f"bearish-decisive:}} $U = {r2['U']:,.0f}$, $p {p2_str}$",
        r"\end{table}",
    ]
    TEX_OUT.write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"Saved LaTeX: {TEX_OUT}")

    # Figure (two-panel violin)
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.titleweight": "bold", "axes.titlesize": 13})

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    palette = {
        "decisive": "#6B7280",
        "ambiguous": "#C0392B",
        "decisive_bullish": "#0EA5E9",
        "decisive_bearish": "#7F1D1D",
    }

    # Left: 2-group violin (decisive pooled vs ambiguous)
    data_L = [
        df_main[df_main["group"].isin(["decisive_bullish", "decisive_neutral",
                                        "decisive_bearish"])]["s"].values,
        df_main[df_main["group"] == "ambiguous"]["s"].values,
    ]
    parts = axL.violinplot(data_L, positions=[0, 1], showmedians=True,
                           showextrema=False, widths=0.78)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor([palette["decisive"], palette["ambiguous"]][i])
        body.set_edgecolor("black"); body.set_alpha(0.78)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.6)
    axL.set_xticks([0, 1])
    axL.set_xticklabels([f"Decisive\n(N={len(data_L[0]):,})",
                         f"Ambiguous\n(N={len(data_L[1]):,})"])
    axL.set_ylabel(r"Learned impact magnitude  $s$")
    axL.set_xlabel("Sentiment confidence group")
    p1_str_short = ("p < 0.001" if r1["p"] < 1e-3 else f"p = {r1['p']:.3f}")
    axL.set_title(f"Ambiguous vs Decisive   (Mann-Whitney  {p1_str_short})")

    # Right: 3-group violin (decisive-bull, decisive-bear, ambiguous)
    data_R = [
        df_main[df_main["group"] == "decisive_bullish"]["s"].values,
        df_main[df_main["group"] == "decisive_bearish"]["s"].values,
        df_main[df_main["group"] == "ambiguous"]["s"].values,
    ]
    parts2 = axR.violinplot(data_R, positions=[0, 1, 2], showmedians=True,
                            showextrema=False, widths=0.78)
    colors_R = [palette["decisive_bullish"], palette["decisive_bearish"],
                palette["ambiguous"]]
    for i, body in enumerate(parts2["bodies"]):
        body.set_facecolor(colors_R[i])
        body.set_edgecolor("black"); body.set_alpha(0.78)
    parts2["cmedians"].set_color("black"); parts2["cmedians"].set_linewidth(1.6)
    axR.set_xticks([0, 1, 2])
    axR.set_xticklabels([
        f"Decisive\nbullish\n(N={len(data_R[0]):,})",
        f"Decisive\nbearish\n(N={len(data_R[1]):,})",
        f"Ambiguous\n(N={len(data_R[2]):,})",
    ])
    axR.set_ylabel(r"Learned impact magnitude  $s$")
    axR.set_xlabel("Sentiment confidence group")
    p2_str_short = ("p < 0.001" if r2["p"] < 1e-3 else f"p = {r2['p']:.3f}")
    axR.set_title(f"Bullish-decisive vs Bearish-decisive   "
                  f"(Mann-Whitney  {p2_str_short})")

    plt.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {FIG_OUT}")


def main():
    print("="*72)
    print("Sentiment vs Event Analysis")
    print("="*72)

    sent = extract_sentiment_scores()
    sl = load_s_lambda_per_window()

    merged = sl.merge(sent, on="id", how="inner")
    print(f"\nMerged rows: {len(merged):,}  "
          f"(unique articles: {merged['id'].nunique():,};  "
          f"windows: {sorted(merged['window'].unique().tolist())})")

    # Robustness: print results at all thresholds, persist threshold=0.60
    df_main = None; r1_main = None; r2_main = None
    for thr in THRESHOLDS:
        df_thr = classify_by_threshold(merged, thr)
        r1, r2 = run_stats_block(df_thr, thr)
        if abs(thr - 0.60) < 1e-9:
            df_main, r1_main, r2_main = df_thr, r1, r2

    # ── Verification checks (the spec's three "assumptions") ──────────────
    if df_main is not None:
        amb = df_main[df_main["group"] == "ambiguous"]["s"].values
        dec = df_main[np.isin(df_main["group"], ["decisive_bullish",
                              "decisive_neutral", "decisive_bearish"])]["s"].values
        bull = df_main[df_main["group"] == "decisive_bullish"]["s"].values
        bear = df_main[df_main["group"] == "decisive_bearish"]["s"].values
        print("\n=== Verification (threshold 0.60) ===")
        v1 = np.mean(amb) > np.mean(dec)
        v2 = r1_main["p"] < 0.001
        v3 = r2_main["p"] > 0.05
        print(f"  Mean s (ambiguous) > Mean s (decisive):       {v1}  "
              f"({np.mean(amb):.4f} vs {np.mean(dec):.4f})")
        print(f"  Mann-Whitney p(amb vs dec) < 0.001:           {v2}  "
              f"(p = {r1_main['p']:.3e})")
        print(f"  Bullish vs Bearish non-significant (p>0.05):  {v3}  "
              f"(p = {r2_main['p']:.3e})")

        write_outputs(df_main, r1_main, r2_main)


if __name__ == "__main__":
    main()
