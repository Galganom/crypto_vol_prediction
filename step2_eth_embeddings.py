"""
Step 2 (ETH): News Embedding Extraction for Ethereum
- Filter news to ETH only
- Extract embeddings using ProsusAI/finbert (768-dim)
- Save embeddings + timestamps as .npz
"""

import pandas as pd
import numpy as np
import os
import torch
from datetime import datetime

df = pd.read_parquet("data/news/processed_news.parquet")
eth = df[df["asset"] == "ETH"].reset_index(drop=True)
print(f"ETH news articles: {len(eth):,}")
print(f"Date range: {datetime.utcfromtimestamp(eth['unix_timestamp'].min())} → "
      f"{datetime.utcfromtimestamp(eth['unix_timestamp'].max())}")

from sentence_transformers import SentenceTransformer

texts = eth["cleaned_text_truncated"].tolist()

print("\nLoading ProsusAI/finbert...")
model = SentenceTransformer("ProsusAI/finbert")
batch_size = 64

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using finbert on {device}")
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

os.makedirs("data/processed", exist_ok=True)
timestamps = eth["unix_timestamp"].values.astype(np.int64)
np.savez("data/processed/eth_news_embeddings.npz",
         embeddings=embeddings, timestamps=timestamps)
print(f"Saved to data/processed/eth_news_embeddings.npz")
print(f"  embeddings: {embeddings.shape}")
print(f"  timestamps: {timestamps.shape}")
