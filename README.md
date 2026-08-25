# Crypto Volatility Prediction

Main expanding-window experiment for 1h and 4h crypto realized volatility.

The production model is now the CryptoBERT-PCA news model:

- Time-series MLP baseline branch trained first.
- News branch uses per-window PCA(256) of CryptoBERT article embeddings.
- Each article gets its own learned impact magnitude and exponential decay rate.
- Reactive price-report headlines are filtered out of the news lookback.
- Final prediction is a multiplicative residual correction to the TS MLP baseline.
- Expanding-window labels are `1..6`, where window `1` is the 2023-01-01 to 2023-06-30 test split.

## Main Benchmark

Each BTC/ETH expanding-window run compares:

- `HAR`
- `EGARCH`
- `EGARCH+NED` = EGARCH corrected by a separately trained CryptoBERT-PCA news branch
- `TS MLP`
- `Main Model` = CryptoBERT-PCA news residual model

Each model is trained separately for `1h` (`RV_1h`) and `4h` (`RV_4h`).

The main model uses P2 as the terminal checkpoint:

| Phase | Trains | LR |
|---|---|---:|
| P1 | TS MLP only | `1e-4` |
| P2 | News residual only | `1e-4` |

## Run

```bash
# BTC full expanding-window benchmark + DM tests
python expanding_window_btc.py

# ETH full expanding-window benchmark + DM tests
python expanding_window_eth.py

```

Useful options:

```bash
# Train without running DM tests at the end
python expanding_window_btc.py --skip-dm

# Re-run DM tests only from saved prediction files
python expanding_window_btc.py --dm-only
python expanding_window_eth.py --dm-only

```

## Outputs

BTC outputs go to `results/main/btc_cryptobert/`.

ETH outputs go to `results/main/eth_cryptobert/`.

Ablation outputs go under `results/ablations/`, separate from the main benchmark results.

Each run writes:

- `expanding_window_metrics.csv`: per-window benchmark metrics.
- `expanding_window_mean_metrics.csv`: arithmetic mean metrics across windows.
- `dm_test_results.csv`: per-window rows plus a `window=mean` summary row by horizon.
- `predictions/window_{n}_{horizon}_predictions.npz`: actuals plus all model forecasts.
- `checkpoints/`: main model, TS MLP, and EGARCH+NED checkpoints.
- `metadata/`: expanding-window indices, normalization stats, and CryptoBERT PCA metadata.

The `mean` row reports arithmetic mean metrics across expanding windows, so `R2` is not inflated by pooling different volatility regimes. The DM statistics on that row are still computed from concatenated loss differences rather than averaged p-values.

## Data Setup

```bash
# BTC FinBERT embeddings/sentiment inputs
python step2_news_embeddings.py

# ETH FinBERT embeddings/sentiment inputs
python step2_eth_embeddings.py

# BTC/ETH CryptoBERT embedding inputs
python step2_eth_cryptobert.py

# BTC topic/category flags
python step3_topic_flags.py
```

ETH topic/category flags are created automatically by `expanding_window_eth.py` if `data/processed/eth_topic_flags.npz` does not exist.

## Repository Layout

```text
main_model_cryptobert.py   # CryptoBERT-PCA main model and training loop
main_model_common.py       # shared data, metrics, baselines, and DM utilities
expanding_window_btc.py    # BTC entrypoint
expanding_window_eth.py    # ETH entrypoint
step*.py                   # data preparation scripts
ablations/                 # current ablation scripts only
```

Run ablation scripts from the repo root, for example:

```bash
python ablations/btc_all6_p2_feature_compare.py
python ablations/btc_all6_p2_feature_compare.py
```

The BTC ablation trains one shared P1 TS branch per window and seed, then reuses
that exact TS state for all static and learned-decay news variants.
