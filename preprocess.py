"""
Crypto News Preprocessing Pipeline
====================================
Single end-to-end script: load CSVs -> dedup -> timestamps -> text cleaning
-> quality filtering -> text length analysis + truncation -> source
normalization -> save parquet.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "csv"
OUT_DIR = BASE_DIR / "processed"
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "processed_news.parquet"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load CSVs
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: Loading CSVs")
print("=" * 70)

frames = []
for path in sorted(CSV_DIR.glob("*_news.csv")):
    asset = path.stem.replace("_news", "")
    df = pd.read_csv(path, low_memory=False)
    df["asset"] = asset
    frames.append(df)

raw = pd.concat(frames, ignore_index=True)
print(f"  Loaded {len(frames)} asset files")
print(f"  Starting rows: {len(raw):,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Deduplication (within each asset)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("STEP 2: Deduplication")
print("=" * 70)

# --- Pass 1: Exact guid dedup within each asset ---
before_exact = len(raw)
raw = raw.sort_values("upvotes", ascending=False)
deduped_exact = raw.drop_duplicates(subset=["guid", "asset"], keep="first")
removed_exact = before_exact - len(deduped_exact)

print(f"\n  Pass 1 — Exact guid dedup (within each asset)")
print(f"    Removed: {removed_exact:,}  |  Remaining: {len(deduped_exact):,}")
print(f"\n    Remaining per asset:")
print(deduped_exact.groupby("asset").size().sort_values(ascending=False).to_string())


# --- Pass 2: Near-duplicate removal per asset via TF-IDF on title ---
def find_near_dupes(df_asset, sim_thresh=0.85, window_seconds=3600):
    """
    Within a single asset's dataframe:
    - Sort by timestamp
    - For each pair with cosine(title) > sim_thresh AND |dt| < window_seconds,
      mark the lower-upvote row (or later row if tied) as a duplicate.
    Returns a boolean mask: True = keep, False = drop.
    """
    df = df_asset.copy().reset_index(drop=True)
    n = len(df)
    if n < 2:
        return pd.Series(True, index=df_asset.index)

    df_sorted = df.sort_values("published_on").reset_index()  # 'index' = original positional index
    timestamps = df_sorted["published_on"].values
    titles = df_sorted["title"].fillna("").values
    upvotes = df_sorted["upvotes"].values

    # TF-IDF on titles
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000)
    try:
        tfidf_matrix = tfidf.fit_transform(titles)
    except ValueError:
        return pd.Series(True, index=df_asset.index)

    drop_positions = set()

    for i in range(n):
        if i in drop_positions:
            continue
        window_end = timestamps[i] + window_seconds
        js = np.where(
            (timestamps > timestamps[i]) & (timestamps <= window_end)
        )[0]
        if len(js) == 0:
            continue

        sims = cosine_similarity(tfidf_matrix[i], tfidf_matrix[js])[0]
        for k, j in enumerate(js):
            if j in drop_positions:
                continue
            if sims[k] >= sim_thresh:
                if upvotes[j] > upvotes[i]:
                    drop_positions.add(i)
                    break
                else:
                    drop_positions.add(j)

    keep_positions = set(range(n)) - drop_positions
    keep_orig_indices = df_sorted.loc[list(keep_positions), "index"].values
    mask = pd.Series(False, index=df_asset.index)
    mask.iloc[keep_orig_indices] = True
    return mask


print(f"\n  Pass 2 — TF-IDF near-dedup (cosine > 0.85, within 1h, per asset)")

asset_groups = deduped_exact.groupby("asset", group_keys=False)
total_near_removed = 0
near_removed_per_asset = {}
keep_masks = []

for asset, group in asset_groups:
    mask = find_near_dupes(group)
    removed = (~mask).sum()
    near_removed_per_asset[asset] = removed
    total_near_removed += removed
    keep_masks.append(group[mask])

deduped = pd.concat(keep_masks, ignore_index=True)

print(f"\n    Near-dupes removed per asset:")
for asset, cnt in sorted(near_removed_per_asset.items(), key=lambda x: -x[1]):
    print(f"      {asset}: {cnt}")

print(f"\n    Total near-dupes removed: {total_near_removed:,}")
print(f"    Remaining after both passes: {len(deduped):,}")

print(f"\n  Dedup summary:")
print(f"    Original rows:          {before_exact:,}")
print(f"    After exact guid dedup: {before_exact - removed_exact:,}  (-{removed_exact:,})")
print(f"    After near-dedup:       {len(deduped):,}  (-{total_near_removed:,})")
print(f"    Total removed:          {before_exact - len(deduped):,}")

df = deduped


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Timestamp Processing
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("STEP 3: Timestamp Processing")
print("=" * 70)

df["unix_timestamp"] = df["published_on"].astype(np.int64)
df["timestamp"] = pd.to_datetime(df["unix_timestamp"], unit="s", utc=True)
df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour

df = df.sort_values("timestamp").reset_index(drop=True)

NOW_TS = int(pd.Timestamp("2026-03-19", tz="UTC").timestamp())
MIN_VALID = int(pd.Timestamp("2015-01-01", tz="UTC").timestamp())

flag_future = df[df["unix_timestamp"] > NOW_TS]
flag_pre2015 = df[df["unix_timestamp"] < MIN_VALID]
flag_exact_dup = df[df.duplicated(subset=["unix_timestamp", "asset"], keep=False)]

print(f"  Future dates (> 2026-03-19):  {len(flag_future):,}")
print(f"  Pre-2015 dates:               {len(flag_pre2015):,}")
print(f"  Exact same second + asset:    {len(flag_exact_dup):,}")
print(f"  Overall date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
print(f"  Total rows: {len(df):,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Text Cleaning
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("STEP 4: Text Cleaning")
print("=" * 70)


def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    # HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # HTML entities
    text = re.sub(r"&[a-zA-Z]+;|&#\d+;", " ", text)
    # URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    # Non-ASCII
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Excessive whitespace / newlines
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


empty_body = df["body"].isna() | (df["body"].str.strip() == "")
print(f"  Rows with empty body before cleaning: {empty_body.sum():,}")

df["_title_clean"] = df["title"].apply(clean_text)
df["_body_clean"] = df["body"].apply(clean_text)

df["cleaned_text"] = np.where(
    df["_body_clean"] == "",
    df["_title_clean"],
    df["_title_clean"] + " " + df["_body_clean"],
)

df.drop(columns=["_title_clean", "_body_clean"], inplace=True)

had_html = df["body"].fillna("").str.contains(r"<[a-zA-Z]", regex=True)
had_url = df["body"].fillna("").str.contains(r"https?://", regex=True)
had_nonascii = df["body"].fillna("").str.contains(r"[^\x00-\x7F]", regex=True)

print(f"  Rows with HTML tags in body:     {had_html.sum():,}")
print(f"  Rows with URLs in body:          {had_url.sum():,}")
print(f"  Rows with non-ASCII in body:     {had_nonascii.sum():,}")

df["char_count"] = df["cleaned_text"].str.len()
df["word_count"] = df["cleaned_text"].str.split().str.len()

print(f"  Empty after cleaning:  {(df['char_count'] == 0).sum():,}")
print(f"  Char count -- mean: {df['char_count'].mean():.0f}  median: {df['char_count'].median():.0f}  max: {df['char_count'].max():,}")
print(f"  Word count -- mean: {df['word_count'].mean():.0f}  median: {df['word_count'].median():.0f}  max: {df['word_count'].max():,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Quality Filtering
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("STEP 5: Quality Filtering (R1, R3, R4, R5, R6 -- no R2 sponsored)")
print("=" * 70)

title_l = df["title"].str.lower().fillna("")
tags_l = df["tags"].fillna("").str.lower()
cats_l = df["categories"].fillna("").str.lower()

R1 = df["char_count"] < 20

price_ticker_pat = (
    r"^bitcoin price[:\s]"
    r"|^btc/usd\b"
    r"|^eth/usd\b"
    r"|^bitcoin price watch"
    r"|^ether price watch"
    r"|\bprice watch\b"
    r"|markets daily"
)
R3 = title_l.str.contains(price_ticker_pat, regex=True, na=False) & (df["word_count"] < 100)

R4 = (
    tags_l.str.contains(r"coindesk indices|coindesk 20", regex=True, na=False)
    & ~tags_l.str.contains(r"news|markets news|analysis", regex=True, na=False)
    & (df["word_count"] < 80)
)

R5 = tags_l.str.contains(r"\bpodcasts?\b", regex=True, na=False) & (df["word_count"] < 60)

R6 = (df["word_count"] < 30) & ~R1

remove_mask = R1 | R3 | R4 | R5 | R6

print(f"  Rows removed per rule:")
for name, mask in [
    ("R1_too_short", R1),
    ("R3_price_ticker", R3),
    ("R4_indices_stub", R4),
    ("R5_podcast_stub", R5),
    ("R6_short_body", R6),
]:
    print(f"    {name:<22} {mask.sum():>5,}")
print(f"    {'TOTAL (union)':<22} {remove_mask.sum():>5,}")

before_filter = len(df)
df = df[~remove_mask].reset_index(drop=True)

print(f"\n  Before: {before_filter:,}")
print(f"  After:  {len(df):,}  (-{before_filter - len(df):,})")

print(f"\n  Remaining rows per asset:")
print(df.groupby("asset").size().sort_values(ascending=False).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Text Length Analysis + Truncation
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("STEP 6: Text Length Analysis + Truncation")
print("=" * 70)

print("\n  Character count distribution:")
print(df["char_count"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(0).to_string())

print("\n  Word count distribution:")
print(df["word_count"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(0).to_string())

bins = [0, 100, 200, 400, 800, 1500, 3000, np.inf]
labels = ["<100", "100-200", "200-400", "400-800", "800-1500", "1500-3000", ">3000"]
wc_bucket = pd.cut(df["word_count"], bins=bins, labels=labels)
print("\n  Word count buckets:")
bucket_counts = wc_bucket.value_counts().sort_index()
for label, cnt in bucket_counts.items():
    pct = cnt / len(df) * 100
    print(f"    {label:<12}  {cnt:>6,}  ({pct:>5.1f}%)")

over_400 = (df["word_count"] > 400).sum()
print(f"\n  Articles > 400 words (need truncation): {over_400:,}  ({over_400 / len(df) * 100:.1f}%)")

# Sentence splitter
ABBREV = re.compile(
    r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|Inc|Ltd|Corp|Co"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\."
)


def split_sentences(text):
    text = ABBREV.sub(lambda m: m.group().replace(".", "<DOT>"), text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.replace("<DOT>", ".") for p in parts if p.strip()]


def truncate(text, max_words=400):
    words = text.split()
    if len(words) <= max_words:
        return text
    sentences = split_sentences(text)
    if len(sentences) <= 3:
        return " ".join(words[:max_words])
    first_two = " ".join(sentences[:2])
    last_one = sentences[-1]
    result = first_two + " [...] " + last_one
    result_words = result.split()
    if len(result_words) > max_words + 50:
        result = " ".join(result_words[:max_words])
    return result


df["cleaned_text_truncated"] = df.apply(
    lambda r: truncate(r["cleaned_text"]) if r["word_count"] > 400 else r["cleaned_text"],
    axis=1,
)

df["truncated_word_count"] = df["cleaned_text_truncated"].str.split().str.len()

was_truncated = df["word_count"] > 400
print(f"\n  Articles truncated: {was_truncated.sum():,}")
if was_truncated.sum() > 0:
    print(f"  Word count BEFORE (truncated articles) -- mean: {df.loc[was_truncated, 'word_count'].mean():.0f}  median: {df.loc[was_truncated, 'word_count'].median():.0f}  max: {df.loc[was_truncated, 'word_count'].max():,}")
    print(f"  Word count AFTER  (truncated articles) -- mean: {df.loc[was_truncated, 'truncated_word_count'].mean():.0f}  median: {df.loc[was_truncated, 'truncated_word_count'].median():.0f}  max: {df.loc[was_truncated, 'truncated_word_count'].max():,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Source Normalization
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("STEP 7: Source Normalization")
print("=" * 70)

df["source"] = df["source"].str.lower().str.strip()

print(f"  Source breakdown:")
print(df["source"].value_counts().head(20).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Assemble Final Output + Save Parquet
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("STEP 8: Final Output")
print("=" * 70)

out_cols = [
    "id", "guid", "timestamp", "unix_timestamp", "date",
    "source", "asset", "title", "body", "cleaned_text", "cleaned_text_truncated",
    "tags", "categories", "upvotes", "downvotes", "char_count", "word_count",
]

missing = [c for c in out_cols if c not in df.columns]
if missing:
    print(f"  WARNING -- missing columns: {missing}")
else:
    print(f"  All {len(out_cols)} output columns present.")

final = df[out_cols].sort_values("timestamp").reset_index(drop=True)

print(f"\n  Total rows:        {len(final):,}")
print(f"  Date range:        {final['timestamp'].min().date()} -> {final['timestamp'].max().date()}")

total_days = (final["timestamp"].max() - final["timestamp"].min()).days
if total_days > 0:
    avg_per_day = len(final) / total_days
    print(f"  Avg news per day:  {avg_per_day:.1f}")

print(f"\n  Per-asset count:")
asset_counts = final.groupby("asset").size().sort_values(ascending=False)
for asset, cnt in asset_counts.items():
    print(f"    {asset:<6}  {cnt:>6,}")

print(f"\n  Source breakdown:")
print(f"    {final['source'].value_counts().to_string()}")

print(f"\n  char_count  -- mean: {final['char_count'].mean():.0f}  median: {final['char_count'].median():.0f}")
print(f"  word_count  -- mean: {final['word_count'].mean():.0f}  median: {final['word_count'].median():.0f}")

print(f"\n  Null counts in final output:")
nulls = final.isnull().sum()
nulls = nulls[nulls > 0]
if len(nulls) == 0:
    print("    None -- clean.")
else:
    print(nulls.to_string())

# Save
final.to_parquet(OUT_PATH, index=False, engine="pyarrow")

size_mb = OUT_PATH.stat().st_size / 1024 / 1024
print(f"\n  Saved -> {OUT_PATH}")
print(f"  File size: {size_mb:.1f} MB")


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL: BTC-Specific Statistics
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("OPTIONAL: BTC-Specific Statistics")
print("=" * 70)

btc = final[final["asset"] == "BTC"].copy()
print(f"  BTC articles: {len(btc):,}")

if len(btc) > 0:
    # Date coverage
    print(f"\n  --- Date Coverage ---")
    print(f"  Range:  {btc['timestamp'].min().date()} -> {btc['timestamp'].max().date()}")
    total_days_btc = (btc["timestamp"].max() - btc["timestamp"].min()).days
    covered = btc["date"].nunique()
    if total_days_btc > 0:
        print(f"  Total calendar days in range: {total_days_btc:,}")
        print(f"  Days with at least 1 article: {covered:,}  ({covered / total_days_btc * 100:.1f}%)")

    # Missing days
    all_dates = pd.date_range(btc["timestamp"].min().date(), btc["timestamp"].max().date(), freq="D")
    covered_set = set(btc["date"])
    missing_days = [d.date() for d in all_dates if d.date() not in covered_set]
    print(f"  Days with ZERO articles:      {len(missing_days):,}")
    if missing_days:
        print(f"  First few missing: {missing_days[:5]}")
        print(f"  Last few missing:  {missing_days[-5:]}")

    # Daily volume
    daily = btc.groupby("date").size()
    print(f"\n  --- Daily Article Volume ---")
    print(f"  Mean:   {daily.mean():.1f} articles/day")
    print(f"  Median: {daily.median():.1f}")
    print(f"  Std:    {daily.std():.1f}")
    print(f"  Min:    {daily.min()}  on {daily.idxmin()}")
    print(f"  Max:    {daily.max()}  on {daily.idxmax()}")

    print(f"\n  Yearly volume:")
    btc["year"] = btc["timestamp"].dt.year
    yearly = btc.groupby("year").agg(
        articles=("id", "count"),
        days_covered=("date", "nunique"),
    ).assign(avg_per_day=lambda x: (x["articles"] / x["days_covered"]).round(1))
    print(yearly.to_string())

    # Hour-of-day distribution
    btc["hour"] = btc["timestamp"].dt.hour
    print(f"\n  --- Hour-of-Day Distribution (UTC) ---")
    hourly = btc["hour"].value_counts().sort_index()
    peak_h = hourly.idxmax()
    print(f"  Peak publishing hour (UTC): {peak_h}:00  ({hourly[peak_h]:,} articles)")
    print(f"  Quietest hour (UTC):        {hourly.idxmin()}:00  ({hourly.min():,} articles)")

    def period(h):
        if 0 <= h < 6:
            return "00-06 (night)"
        if 6 <= h < 12:
            return "06-12 (morning)"
        if 12 <= h < 18:
            return "12-18 (afternoon)"
        return "18-24 (evening)"

    btc["period"] = btc["hour"].apply(period)
    print(f"\n  By 6-hour period:")
    print(btc["period"].value_counts().sort_index().to_string())

    # Engagement
    print(f"\n  --- Engagement (upvotes/downvotes) ---")
    print(f"  Upvotes   -- mean: {btc['upvotes'].mean():.2f}  median: {btc['upvotes'].median():.0f}  max: {btc['upvotes'].max()}")
    print(f"  Downvotes -- mean: {btc['downvotes'].mean():.2f}  median: {btc['downvotes'].median():.0f}  max: {btc['downvotes'].max()}")

    # Coverage during known high-volatility periods
    print(f"\n  --- Coverage During Known High-Volatility Periods ---")
    vol_events = {
        "COVID crash (Mar 2020)": ("2020-03-01", "2020-03-31"),
        "Bull run peak (Nov 2021)": ("2021-10-01", "2021-12-31"),
        "LUNA collapse (May 2022)": ("2022-05-01", "2022-06-15"),
        "FTX collapse (Nov 2022)": ("2022-11-01", "2022-11-30"),
        "ETF approval rally (Jan 2024)": ("2024-01-01", "2024-01-31"),
        "BTC ATH (Mar 2024)": ("2024-03-01", "2024-03-31"),
    }
    btc["date_dt"] = pd.to_datetime(btc["date"])
    for event, (start, end) in vol_events.items():
        mask = (btc["date_dt"] >= start) & (btc["date_dt"] <= end)
        n = mask.sum()
        days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
        print(f"    {event:<40} {n:>5,} articles  (~{n / days:.1f}/day)")

print(f"\nDone.")
