"""Cleaner BTC news category labels.

Three fixes over the existing DeBERTa-zero-shot labels:
  (1) Drop labels with confidence < CONF_THRESH and reassign to "Other".
  (2) Regex pre-filter for recurring editorial series → "Newsletter_Series".
      These are columns/newsletters that the zero-shot model often miscategorizes
      because of stock framing words ("advisors", "professionalization", etc).
  (3) Re-label remaining articles with a STRONGER zero-shot model
      (MoritzLaurer/deberta-v3-large-zeroshot-v2.0) — better generalization on
      financial/crypto text vs the original.

Output:
  data/processed/btc_news_categories_clean.npz
    keys: category, confidence, label_names, source, unix_timestamp
    where source ∈ {"editorial_regex", "deberta_v3_large", "low_conf_other"}
  data/processed/btc_news_categories_clean.csv (same data, plus title for inspection)
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ─── Config ──────────────────────────────────────────────────────────────────
NEWS_PATH = REPO_ROOT / "data/news/processed_news.parquet"
OUT_NPZ = REPO_ROOT / "data/processed/btc_news_categories_clean.npz"
OUT_CSV = REPO_ROOT / "data/processed/btc_news_categories_clean.csv"

# Stronger zero-shot classifier list (ranked by typical zero-shot benchmark perf).
# We try each in order; first one that loads is used.
MODEL_CANDIDATES = [
    "MoritzLaurer/ModernBERT-large-zeroshot-v2.0",      # ModernBERT backbone — strongest 2025-era zero-shot
    "MoritzLaurer/bge-m3-zeroshot-v2.0",                # BGE-M3 backbone — multilingual + strong
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",      # prev baseline (fallback)
]
BATCH_SIZE = 8
MAX_LEN = 512
CONF_THRESH = 0.25   # lowered further for 14-way softmax

# Event-typology taxonomy aligned with academic crypto-news studies
# (Karalevicius 2018, Liu & Tsyvinski 2021, RavenPack News Analytics):
# We categorize by the *underlying triggering event*, not the surface vocabulary.
LABEL_NAMES = [
    "Macro",
    "Geopolitical",
    "Regulatory_Legal",
    "Security_Crime",
    "Institutional_Flows",
    "Protocol_Infrastructure",
    "OnChain_Flows",
    "Product_Adoption",
    "Market_Commentary",
    "Opinion_Editorial",
    "Other",
]

# Event-focused hypothesis template:
HYPOTHESIS = "This article reports on a {} event affecting cryptocurrency markets."
LABEL_DESCRIPTIONS = {
    "Macro": "macroeconomic release such as CPI, jobs report, GDP, FOMC meeting, Federal Reserve interest-rate decision, central bank policy, or inflation data",
    "Geopolitical": "geopolitical event such as a war, sanctions, election outcome, country-level ban, martial law, tariffs, or sovereign action that affects financial markets",
    "Regulatory_Legal": "regulatory action, SEC or CFTC lawsuit, enforcement, court ruling, legislation, ETF approval as a regulatory milestone, or government policy change",
    "Security_Crime": "exchange hack, smart-contract exploit, ransomware, fraud, theft, scam, money laundering charge, or arrest of a crypto operator",
    "Institutional_Flows": "ETF inflow or outflow, corporate Bitcoin treasury purchase by a public company like MicroStrategy, BlackRock or Fidelity fund flow, institutional allocation, or country-level adoption",
    "Protocol_Infrastructure": "blockchain protocol upgrade, hard fork, halving, Lightning Network change, mining difficulty or hash-rate event, or developer activity affecting Bitcoin or Ethereum",
    "OnChain_Flows": "on-chain or exchange flow event such as whale movement, long-term holder transfer, exchange reserve change, wallet flows, large liquidation cascade, exchange outage, proof-of-reserves disclosure, or CryptoQuant or Glassnode metric release",
    "Product_Adoption": "new product launch, wallet integration, payment-rail partnership, merchant adoption, or commercial product release",
    "Market_Commentary": "backward-looking analysis of crypto price action that does not report any new triggering event, such as technical analysis, chart pattern commentary, price-target columns, or daily market recaps that only describe moves already in progress",
    "Opinion_Editorial": "opinion essay, op-ed, philosophical commentary, or thought-leadership argument about the long-run future of crypto",
    "Other": "celebrity gossip, sponsored content, awards, conferences, or topics that do not fit any of the above event types",
}

# ─── Editorial recurring-series regex ────────────────────────────────────────
EDITORIAL_PATTERNS = [
    r"^Crypto for Advisors\b",
    r"^Protocol Village\b",
    r"^Markets Daily\b",
    r"^Markets?\s+Today\b",                # NEW: Markets Today / Market Today (2025 variant of Markets Daily)
    r"^First Mover\b",
    r"^Crypto Long & Short\b",
    r"^The Node\b",
    r"^The Protocol\b",                    # NEW: 2025 weekly protocol recap series
    r"^Money Reimagined\b",
    r"^State of Crypto\b",
    r"^What['’]s Hot\b",
    r"^Coindesk\s+\d{1,3}\b",
    r"^CoinDesk\s+\d{1,3}\b",
    r"^Coindesk Indices\b",
    r"^CoinDesk Indices\b",
    r"^CoinDesk 20 Performance Update\b",  # NEW: 2025 daily CoinDesk-20 price tracker
    r"^Asia Morning Briefing\b",           # NEW: 2025 daily Asia recap
    r"^Crypto Markets? Today\b",           # NEW: 2025 daily wrap (singular + plural)
    r"^Chart of the (Week|Day|Month)\b",   # NEW: recurring chart series
    r"^(?:CoinDesk )?Weekly Recap\b",      # NEW: weekly recap series
    r"^El Salvador Dispatch\b",            # NEW: country dispatch series
    r"^Best of the Week\b",                # NEW
    r"^Most Influential\b",                # NEW
    r"\bCrypto Daybook\b",
    r"\bMarket\s+Wrap\b",
    r"\bLayer 2\s*\(.*newsletter\)",
    r"^.*\bnewsletter\b.*$",
]
EDITORIAL_RE = re.compile("|".join(EDITORIAL_PATTERNS), flags=re.IGNORECASE)


def is_editorial(title: str) -> bool:
    return bool(EDITORIAL_RE.search(str(title)))


def main():
    print(f"Loading news from {NEWS_PATH}")
    df = pd.read_parquet(NEWS_PATH)
    btc = df[df["asset"] == "BTC"].reset_index(drop=True)
    print(f"BTC articles: {len(btc):,}")

    # Step 2: regex prefilter
    edit_mask = btc["title"].apply(is_editorial)
    n_edit = int(edit_mask.sum())
    print(f"Editorial-regex matches: {n_edit:,} ({n_edit/len(btc)*100:.1f}%)")

    # Pre-allocate outputs
    out_cat = np.empty(len(btc), dtype=object)
    out_conf = np.zeros(len(btc), dtype=np.float32)
    out_source = np.empty(len(btc), dtype=object)
    out_cat[edit_mask.values] = "Newsletter_Series"
    out_conf[edit_mask.values] = 1.0
    out_source[edit_mask.values] = "editorial_regex"

    # Step 3: stronger zero-shot for everything else
    todo_idx = np.where(~edit_mask.values)[0]
    device = 0 if torch.cuda.is_available() else -1
    clf = None
    chosen_model = None
    for cand in MODEL_CANDIDATES:
        try:
            print(f"Trying zero-shot model: {cand}")
            clf = pipeline("zero-shot-classification", model=cand,
                           device=device, batch_size=BATCH_SIZE)
            chosen_model = cand
            print(f"  → loaded {cand}")
            break
        except Exception as e:
            print(f"  ✗ {cand} failed: {type(e).__name__}: {str(e)[:200]}")
    if clf is None:
        raise RuntimeError("No zero-shot model could be loaded from MODEL_CANDIDATES")
    print(f"Re-labeling {len(todo_idx):,} non-editorial articles with {chosen_model}...")

    label_descriptions = [LABEL_DESCRIPTIONS[lab] for lab in LABEL_NAMES]
    label_to_idx = {lab: i for i, lab in enumerate(LABEL_NAMES)}

    texts = btc.loc[todo_idx, "cleaned_text_truncated"].astype(str).tolist()
    # Truncate to MAX_LEN words to keep tokenization fast & avoid OOMs
    texts = [" ".join(t.split()[:MAX_LEN]) for t in texts]

    n_done = 0
    for batch_start in tqdm(range(0, len(texts), BATCH_SIZE), desc="zero-shot"):
        batch = texts[batch_start:batch_start + BATCH_SIZE]
        out = clf(
            batch,
            candidate_labels=label_descriptions,
            hypothesis_template=HYPOTHESIS,
            multi_label=False,
        )
        if isinstance(out, dict):
            out = [out]
        for i, res in enumerate(out):
            global_pos = todo_idx[batch_start + i]
            top_desc = res["labels"][0]
            top_score = float(res["scores"][0])
            # Map back from description to canonical label
            top_label = LABEL_NAMES[label_descriptions.index(top_desc)]
            # Step 1: confidence threshold
            if top_score < CONF_THRESH:
                out_cat[global_pos] = "Other"
                out_conf[global_pos] = top_score
                out_source[global_pos] = "low_conf_other"
            else:
                out_cat[global_pos] = top_label
                out_conf[global_pos] = top_score
                out_source[global_pos] = "deberta_v3_large"
        n_done += len(batch)

    # Save
    out_dir = OUT_NPZ.parent; out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_NPZ,
        category=out_cat.astype(object),
        confidence=out_conf,
        source=out_source.astype(object),
        unix_timestamp=btc["unix_timestamp"].values.astype(np.int64),
        label_names=np.array(LABEL_NAMES + ["Newsletter_Series"], dtype=object),
    )
    print(f"Saved: {OUT_NPZ}")

    # Sanity-check distribution
    out_df = pd.DataFrame({
        "id": btc["id"], "unix_timestamp": btc["unix_timestamp"], "title": btc["title"],
        "category": out_cat, "confidence": out_conf, "source": out_source,
    })
    out_df.to_csv(OUT_CSV, index=False, float_format="%.6f")
    print(f"Saved: {OUT_CSV}")
    print("\nNew category distribution:")
    print(out_df["category"].value_counts(normalize=False).to_string())
    print("\nLabel sources:")
    print(out_df["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
