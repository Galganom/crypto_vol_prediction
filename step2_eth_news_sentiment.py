"""Generate FinBERT sentiment probabilities for Ethereum news."""

import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32
MAX_LENGTH = 512

NEWS_PATH = "data/news/processed_news.parquet"
EMBEDDING_PATH = "data/processed/eth_news_embeddings.npz"
OUTPUT_PATH = "data/processed/eth_news_sentiment.npz"


def main():
    news = pd.read_parquet(NEWS_PATH)
    eth = news[news["asset"] == "ETH"].reset_index(drop=True)

    texts = eth["cleaned_text_truncated"].astype(str).tolist()
    timestamps = eth["unix_timestamp"].to_numpy(dtype=np.int64)

    # 감성 결과가 기존 ETH 임베딩과 같은 기사 순서인지 확인
    with np.load(EMBEDDING_PATH) as embeddings:
        if not np.array_equal(timestamps, embeddings["timestamps"]):
            raise ValueError(
                "ETH news rows and eth_news_embeddings.npz are not aligned"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"ETH articles: {len(texts):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    ).to(device).eval()

    # 프로젝트에서 기대하는 열 순서:
    # [positive, negative, neutral]
    label2id = {
        str(label).lower(): int(index)
        for label, index in model.config.label2id.items()
    }
    column_order = [
        label2id["positive"],
        label2id["negative"],
        label2id["neutral"],
    ]

    sentiment_probs = np.empty((len(texts), 3), dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, len(texts), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(texts))

            encoded = tokenizer(
                texts[start:end],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
            probs = probs[:, column_order]

            sentiment_probs[start:end] = (
                probs.cpu().numpy().astype(np.float32)
            )

            if start == 0 or end == len(texts) or start % (BATCH_SIZE * 20) == 0:
                print(f"{end:>6,}/{len(texts):,}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        sentiment_probs=sentiment_probs,
        timestamps=timestamps,
    )

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Shape: {sentiment_probs.shape}")
    print(
        "Probability-sum range:",
        sentiment_probs.sum(axis=1).min(),
        sentiment_probs.sum(axis=1).max(),
    )


if __name__ == "__main__":
    main()
