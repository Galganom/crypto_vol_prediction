"""
Step 2: News Embedding Extraction
- Filter news to BTC only
- Extract embeddings using ProsusAI/finbert (768-dim, financial domain)
- Save embeddings + timestamps as .npz
"""

import pandas as pd
import numpy as np
import os
import torch
from datetime import datetime

# ── Load & filter ─────────────────────────────────────────────────────────────
df = pd.read_parquet("data/news/processed_news.parquet")
btc = df[df["asset"] == "BTC"].reset_index(drop=True)
print(f"BTC news articles: {len(btc):,}")
print(f"Date range: {datetime.utcfromtimestamp(btc['unix_timestamp'].min())} → "
      f"{datetime.utcfromtimestamp(btc['unix_timestamp'].max())}")

# ── Embed ─────────────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer

texts = btc["cleaned_text_truncated"].tolist()

print("\nLoading ProsusAI/finbert...")
model = SentenceTransformer("ProsusAI/finbert")
batch_size = 64
model_name = "finbert"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using model: {model_name} on {device}")
print(f"Encoding {len(texts):,} articles (batch_size={batch_size})...")

embeddings = model.encode(
    texts,
    batch_size=batch_size,
    show_progress_bar=True,
    normalize_embeddings=True,
    device=device,
)
embeddings = np.array(embeddings, dtype=np.float32)

print(f"\nEmbedding shape: {embeddings.shape}")
print(f"Embedding dim: {embeddings.shape[1]}")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
timestamps = btc["unix_timestamp"].values.astype(np.int64)
np.savez("data/processed/btc_news_embeddings.npz",
         embeddings=embeddings, timestamps=timestamps)
print(f"\nSaved to data/processed/btc_news_embeddings.npz")
print(f"  embeddings: {embeddings.shape}")
print(f"  timestamps: {timestamps.shape}")
