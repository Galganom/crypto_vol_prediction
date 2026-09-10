"""
Step 2 (BTC CryptoBERT): Extract CryptoBERT embeddings for BTC news.
Uses AutoModel + AutoTokenizer directly (with explicit truncation) and applies
mean-pooling + L2 normalization to match the BTC CryptoBERT format.
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

OUT_PATH = "data/processed/btc_news_embeddings_cryptobert.npz"
MODEL_NAME = "ElKulako/cryptobert"
MAX_LEN = 512  # within model's max_position_embeddings; truncate longer texts
BATCH_SIZE = 32


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def main():
    df = pd.read_parquet("data/news/processed_news.parquet")
    btc = df[df["asset"] == "BTC"].reset_index(drop=True)
    print(f"BTC news articles: {len(btc):,}")
    print(f"Date range: {datetime.utcfromtimestamp(btc['unix_timestamp'].min())} -> "
          f"{datetime.utcfromtimestamp(btc['unix_timestamp'].max())}")

    texts = btc["cleaned_text_truncated"].astype(str).tolist()
    timestamps = btc["unix_timestamp"].values.astype(np.int64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {MODEL_NAME} (device={device})...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    embs = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    print(f"Encoding with mean-pool + L2-norm, max_len={MAX_LEN}, batch={BATCH_SIZE}...")
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc = tok(batch, padding=True, truncation=True, max_length=MAX_LEN,
                      return_tensors="pt").to(device)
            out = model(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
            normed = F.normalize(pooled, p=2, dim=1)
            embs[i:i + BATCH_SIZE] = normed.cpu().numpy().astype(np.float32)
            if (i // BATCH_SIZE) % 25 == 0:
                print(f"  {i + len(batch):>6,}/{len(texts):,}  norm={np.linalg.norm(embs[i]):.4f}")

    print(f"\nEmbedding shape: {embs.shape}")
    print(f"Embedding norm: mean={np.linalg.norm(embs, axis=1).mean():.4f}, "
          f"std={np.linalg.norm(embs, axis=1).std():.4f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(OUT_PATH, embeddings=embs, timestamps=timestamps)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
