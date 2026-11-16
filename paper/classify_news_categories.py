"""Regenerate the original paper's 12-category post-hoc taxonomy.

Only retained BTC articles eligible for a saved 4h test forecast are labelled.
Uses the original label definitions, editorial rules and confidence threshold.
Results are resumable and keyed to original parquet article rows/timestamps.
"""
from pathlib import Path
import json
import os
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", "/tmp/crypto-paper-revision/huggingface")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/crypto-paper-revision/inductor")
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ablations.relabel_news_categories import LABEL_NAMES, LABEL_DESCRIPTIONS, HYPOTHESIS, EDITORIAL_RE

MODEL = "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"
REVISION = "a51e07b524299e309dd2b88d48b0cfa2bd9ec598"
OUT = ROOT / "results/analysis/paper_revision"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)


def main():
    news = pd.read_parquet(ROOT / "data/news/processed_news.parquet")
    news = news[news.asset == "BTC"].reset_index(drop=True)
    pars = pd.read_csv(ROOT / "paper/generated/article_parameters.csv")
    indices = np.sort(pars[(pars.asset == "BTC") & (pars.horizon == "4h")].article_row.unique())
    path = OUT / "article_categories.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {int(r["article_row"]) for r in rows}
    pending = []
    for i in indices:
        if i in done: continue
        r = news.iloc[i]
        if EDITORIAL_RE.search(str(r.title)):
            rows.append(dict(article_row=int(i), timestamp=int(r.unix_timestamp), category="Newsletter_Series", confidence=1., source="editorial_regex", title=r.title))
        else:
            pending.append(int(i))
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Eligible unique articles={len(indices)}, NLI pending={len(pending)}, existing/editorial={len(rows)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
              dtype=torch.bfloat16, attn_implementation="sdpa", reference_compile=False).cuda().eval()
    entail = model.config.label2id["entailment"]
    hypotheses = [HYPOTHESIS.format(LABEL_DESCRIPTIONS[label]) for label in LABEL_NAMES]
    start = time.monotonic()
    # Four articles = 44 premise/hypothesis pairs. Tokenization truncates the
    # premise only, preserving all label definitions within a 512-token budget.
    for offset in range(0, len(pending), 4):
        batch = pending[offset:offset+4]
        premises, hyps = [], []
        for i in batch:
            text = " ".join(str(news.iloc[i].cleaned_text_truncated).split()[:512])
            premises.extend([text] * len(hypotheses))
            hyps.extend(hypotheses)
        encoded = tokenizer(premises, hyps, padding=True, truncation="only_first", max_length=512, return_tensors="pt")
        logits = []
        with torch.inference_mode():
            for k in range(0, len(premises), 11):
                logits.append(model(**{name: value[k:k+11].cuda() for name, value in encoded.items()}).logits[:, entail].float())
        scores = torch.cat(logits).reshape(len(batch), len(hypotheses)).softmax(-1).cpu().numpy()
        for j, i in enumerate(batch):
            best = int(np.argmax(scores[j])); confidence = float(scores[j, best])
            r = news.iloc[i]
            rows.append(dict(article_row=i, timestamp=int(r.unix_timestamp),
                category=LABEL_NAMES[best] if confidence >= .25 else "Other", confidence=confidence,
                source="modernbert_nli" if confidence >= .25 else "low_conf_other", title=r.title))
        if offset % 80 == 0 or offset + 4 >= len(pending):
            pd.DataFrame(rows).sort_values("article_row").to_csv(path, index=False)
            print(f"NLI {min(offset+4,len(pending))}/{len(pending)}; {time.monotonic()-start:.0f}s", flush=True)
    result = pd.DataFrame(rows).sort_values("article_row")
    assert set(result.article_row) == set(indices) and result.article_row.is_unique
    result.to_csv(path, index=False)
    config = dict(model=MODEL, revision=REVISION, tokenizer_max_length=512,
        article_word_cap=512, precision="bfloat16", attention="sdpa", threshold=.25,
        candidates=LABEL_NAMES, descriptions=LABEL_DESCRIPTIONS, hypothesis_template=HYPOTHESIS,
        editorial_regex=EDITORIAL_RE.pattern, N=len(result), scope="BTC 4h eligible test articles",
        transformers_version=__import__("transformers").__version__, torch_version=torch.__version__)
    (OUT / "category_config.json").write_text(json.dumps(config, indent=2)+"\n")
    print(result.category.value_counts().to_string(), flush=True)
    print("CATEGORY CLASSIFICATION COMPLETE", flush=True)


if __name__ == "__main__": main()
