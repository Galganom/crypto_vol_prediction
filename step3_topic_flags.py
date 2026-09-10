"""
Step 3B: BTC Topic Flag Extraction

Builds a metadata-driven, multi-label topic representation for BTC news.

Rationale:
  BTC news in this corpus is naturally multi-topic. A single zero-shot label
  often collapses stories such as "SEC delays Bitcoin ETF" into one bucket,
  even though the article is simultaneously regulatory and ETF-related.

Recommended topic flags:
  1. Policy/Legal
  2. Macro
  3. ETF/Institutional/Adoption
  4. Exchange/Intermediary
  5. Protocol/Mining
  6. Security/Crime

Outputs:
  data/processed/btc_topic_flags.npz
    - topic_names: (6,) string array
    - topic_scores: (N, 6) int16 weighted rule scores
    - topic_flags: (N, 6) uint8 multi-hot flags
    - primary_topic: (N,) string array
    - match_count: (N,) int16 number of matched topics per article
    - nonreactive_mask: (N,) uint8 mask for the repo's existing reactive filter

  results/metadata/btc_topic_schema_summary.csv
"""

import os
import re
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results/metadata"
OUT_NPZ = f"{PROCESSED_DIR}/btc_topic_flags.npz"
OUT_CSV = f"{RESULTS_DIR}/btc_topic_schema_summary.csv"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg):
    print(msg, flush=True)


TOPIC_RULES = OrderedDict(
    {
        "Policy/Legal": {
            "tags": {
                "regulation",
                "regulations",
                "legal",
                "policy",
                "sec",
                "cftc",
            },
            "categories": {"regulation"},
            "text": [
                r"\bsec\b",
                r"\bcftc\b",
                r"\bdoj\b",
                r"\blawsuit\b",
                r"\bsettlement\b",
                r"\bcourt\b",
                r"\blegal\b",
                r"\bregulat(?:ion|ory)?\b",
                r"\benforcement\b",
                r"\bban(?:ned|s)?\b",
                r"\bbitlicense\b",
            ],
        },
        "Macro": {
            "tags": {
                "federal reserve",
                "inflation",
                "policy",
                "donald trump",
            },
            "categories": {"macroeconomics"},
            "text": [
                r"\bfederal reserve\b",
                r"\bfed\b",
                r"\binflation\b",
                r"\bcpi\b",
                r"\bppi\b",
                r"\binterest rate[s]?\b",
                r"\brate cut[s]?\b",
                r"\brate hike[s]?\b",
                r"\bpowell\b",
                r"\bjobs report\b",
                r"\bpayrolls?\b",
                r"\bcentral bank\b",
                r"\becb\b",
                r"\bboj\b",
                r"\byield[s]?\b",
            ],
        },
        "ETF/Institutional/Adoption": {
            "tags": {
                "bitcoin etf",
                "etf",
                "etfs",
                "microstrategy",
                "blackrock",
                "fidelity",
                "grayscale",
                "payments",
                "banking",
                "investments",
                "crypto for advisors",
                "el salvador",
            },
            "categories": {"digital asset treasury"},
            "text": [
                r"\betf\b",
                r"\bspot bitcoin\b",
                r"\binstitutional\b",
                r"\btreasury\b",
                r"\basset manager\b",
                r"\bblackrock\b",
                r"\bfidelity\b",
                r"\bgrayscale\b",
                r"\bmicrostrategy\b",
                r"\bpayment[s]?\b",
                r"\bmerchant[s]?\b",
                r"\badoption\b",
                r"\bcorporate reserve\b",
            ],
        },
        "Exchange/Intermediary": {
            "tags": {
                "exchanges",
                "coinbase",
                "binance",
                "kraken",
                "stablecoins",
                "tether",
                "derivatives",
                "options",
                "futures",
            },
            "categories": {"exchange"},
            "text": [
                r"\bexchange\b",
                r"\bcoinbase\b",
                r"\bbinance\b",
                r"\bkraken\b",
                r"\bbitfinex\b",
                r"\bcustod(?:y|ian)\b",
                r"\bbroker(?:age)?\b",
                r"\bstablecoin[s]?\b",
                r"\btether\b",
                r"\busdt\b",
                r"\busdc\b",
                r"\bwithdraw(?:al|als)?\b",
                r"\bderivatives?\b",
                r"\bfutures\b",
                r"\boptions\b",
            ],
        },
        "Protocol/Mining": {
            "tags": {
                "bitcoin mining",
                "mining",
                "technology news",
                "technology",
                "tech",
                "defi",
                "the node",
                "lightning network",
                "blockchain",
            },
            "categories": {"mining", "technology", "forks", "blockchain"},
            "text": [
                r"\bmining\b",
                r"\bminer[s]?\b",
                r"\bhashrate\b",
                r"\bhalving\b",
                r"\blightning\b",
                r"\bordinals\b",
                r"\bprotocol\b",
                r"\bfork\b",
                r"\bwallet\b",
                r"\bdefi\b",
                r"\bsidechain\b",
                r"\blayer[ -]?2\b",
            ],
        },
        "Security/Crime": {
            "tags": {
                "crime",
            },
            "categories": set(),
            "text": [
                r"\bhack(?:ed|ing)?\b",
                r"\bexploit(?:ed)?\b",
                r"\bbreach\b",
                r"\bfraud\b",
                r"\bscam\b",
                r"\bransom(?:ware)?\b",
                r"\btheft\b",
                r"\bstolen\b",
                r"\brobbery\b",
                r"\bmoney laundering\b",
                r"\bsecurity failure\b",
            ],
        },
    }
)

PRIMARY_TIEBREAK = [
    "Security/Crime",
    "Macro",
    "Policy/Legal",
    "ETF/Institutional/Adoption",
    "Exchange/Intermediary",
    "Protocol/Mining",
]

REACTIVE_PATTERNS = [
    r"(?i)bitcoin\s+(tops|surges|jumps|climbs|hits|breaks|crosses|retakes|soars|rallies|rises|reclaims)",
    r"(?i)bitcoin\s+(tumbles|slumps|falls|drops|plunges|dips|crashes|sinks|slides|retreats|plummets)",
    r"(?i)bitcoin\s+(back\s+over|back\s+above|back\s+below)",
    r"(?i)^first\s+mover\s+(asia|americas|europe)",
    r"(?i)(price|btc)\s+(climbs|falls|hits|drops|rises|surges|tumbles|soars|plunges)",
    r"(?i)(tops|breaks|crosses|surpasses|reclaims|retakes)\s+\$[\d,]+",
    r"(?i)(below|above|over|past|under|to)\s+\$[\d,.]+k?\b",
    r"(?i)market\s+wrap",
    r"(?i)(new|all[- ]time|record|yearly|weekly|monthly)\s+(high|low)",
    r"(?i)\$[\d,.]+k?\s*(as|after|amid|on|in)",
    r"(?i)^bitcoin\s+\$",
    r"(?i)(striking\s+distance|within\s+reach)",
]


def split_parts(value):
    if pd.isna(value):
        return []
    parts = re.split(r"[|,;/]+", str(value))
    return [p.strip().lower() for p in parts if p.strip()]


def compile_rules():
    compiled = OrderedDict()
    for topic, cfg in TOPIC_RULES.items():
        compiled[topic] = {
            "tags": {x.lower() for x in cfg["tags"]},
            "categories": {x.lower() for x in cfg["categories"]},
            "text": [re.compile(p, re.I) for p in cfg["text"]],
        }
    return compiled


def article_score(row, rules):
    title = str(row.get("title", "") or "")
    body = str(row.get("cleaned_text_truncated", "") or "")
    tags = set(split_parts(row.get("tags", "")))
    cats = set(split_parts(row.get("categories", "")))
    text = " \n ".join([title, body, " ".join(tags), " ".join(cats)])

    scores = []
    for topic, cfg in rules.items():
        score = 0
        score += 3 * sum(tag in tags for tag in cfg["tags"])
        score += 2 * sum(cat in cats for cat in cfg["categories"])
        score += sum(bool(p.search(text)) for p in cfg["text"])
        scores.append(score)
    return np.array(scores, dtype=np.int16)


def primary_from_scores(scores, topic_names):
    if scores.max() <= 0:
        return "Other/Analysis"
    winners = [topic_names[i] for i, score in enumerate(scores) if score == scores.max()]
    if len(winners) == 1:
        return winners[0]
    for topic in PRIMARY_TIEBREAK:
        if topic in winners:
            return topic
    return winners[0]


# Tag-based recap detection (Option C). The publisher's `tags` column reliably
# marks price-recap / TA / recurring market columns; using it gives higher
# recall AND higher precision than the previous title-regex (which over-caught
# macro/mining news matching e.g. `(record|all-time) high`).
RECAP_TAGS_SET = {
    "price-news", "Markets News", "Market Wrap", "Bitcoin Price",
    "Technical Analysis", "First Mover", "Bitcoin Long & Short",
    "Crypto Long & Short", "Markets Daily", "Asia Pacific", "Trading",
}
SAFE_RECAP_TITLE_PATTERNS = [
    r"\bmarket\s+wrap\b",
    r"^(?:bitcoin|btc)\s+\$",
    r"^first\s+mover\s+(asia|americas|europe)",
]
_SAFE_RECAP_RE = re.compile("|".join(SAFE_RECAP_TITLE_PATTERNS), flags=re.IGNORECASE)


def reactive_mask(df):
    """nonreactive_mask: 1 = keep (causal news), 0 = drop (recap/price-action).
    Uses publisher tags first (high precision/recall), then a small high-precision
    title-regex augmentation."""
    tags_col = df["tags"].fillna("").astype(str)
    by_tag = tags_col.apply(
        lambda t: any(tag in [p.strip() for p in t.split("|")] for tag in RECAP_TAGS_SET)
    )
    titles = df["title"].fillna("").astype(str)
    by_title = titles.str.contains(_SAFE_RECAP_RE, regex=True, na=False)
    recap = (by_tag | by_title).values
    return (~recap).astype(np.uint8)


def main():
    log("Loading BTC news...")
    news_df = pd.read_parquet("data/news/processed_news.parquet")
    btc = news_df[news_df["asset"] == "BTC"].reset_index(drop=True)
    log(f"BTC articles: {len(btc):,}")
    log(
        "Date range: "
        f"{datetime.utcfromtimestamp(btc['unix_timestamp'].min())} -> "
        f"{datetime.utcfromtimestamp(btc['unix_timestamp'].max())}"
    )

    topic_names = np.array(list(TOPIC_RULES.keys()), dtype=object)
    rules = compile_rules()
    nonreactive = reactive_mask(btc)

    log("Scoring topic rules...")
    topic_scores = np.stack([article_score(row, rules) for _, row in btc.iterrows()])
    topic_flags = (topic_scores > 0).astype(np.uint8)
    match_count = topic_flags.sum(axis=1).astype(np.int16)
    primary_topic = np.array(
        [primary_from_scores(score_row, topic_names) for score_row in topic_scores],
        dtype=object,
    )

    np.savez(
        OUT_NPZ,
        topic_names=topic_names,
        topic_scores=topic_scores.astype(np.int16),
        topic_flags=topic_flags.astype(np.uint8),
        primary_topic=primary_topic,
        match_count=match_count,
        nonreactive_mask=nonreactive,
    )
    log(f"Saved: {OUT_NPZ}")

    total = len(btc)
    nr_total = int(nonreactive.sum())
    rows = []
    log("\nTopic coverage:")
    for i, topic in enumerate(topic_names):
        n_all = int(topic_flags[:, i].sum())
        n_nr = int(topic_flags[nonreactive.astype(bool), i].sum())
        rows.append(
            {
                "topic": topic,
                "count_all": n_all,
                "share_all": n_all / total,
                "count_nonreactive": n_nr,
                "share_nonreactive": n_nr / max(nr_total, 1),
            }
        )
        log(f"  {topic:<28s} {n_all:>6,} all  |  {n_nr:>6,} non-reactive")

    overlap_rows = []
    overlap_counts = pd.Series(match_count[nonreactive.astype(bool)]).value_counts().sort_index()
    log("\nNon-reactive overlap:")
    for k, v in overlap_counts.items():
        overlap_rows.append(
            {
                "topic": f"match_count_{int(k)}",
                "count_all": np.nan,
                "share_all": np.nan,
                "count_nonreactive": int(v),
                "share_nonreactive": int(v) / max(nr_total, 1),
            }
        )
        log(f"  {int(k)} topic matches: {int(v):>6,} ({int(v)/max(nr_total,1):>5.1%})")

    summary = pd.DataFrame(rows + overlap_rows)
    summary.to_csv(OUT_CSV, index=False)
    log(f"\nSaved: {OUT_CSV}")

    primary_counts = pd.Series(primary_topic[nonreactive.astype(bool)]).value_counts()
    log("\nPrimary topic distribution (non-reactive):")
    for topic, count in primary_counts.items():
        log(f"  {topic:<28s} {int(count):>6,} ({int(count)/max(nr_total,1):>5.1%})")

    log("\nDone.")


if __name__ == "__main__":
    main()
