"""Replay all saved NED test forecasts from price features and article inputs.

Complements verify_price_pipeline.py: this checks the complete news correction
for both assets, both horizons, and all six windows, without training a model.
"""
from pathlib import Path
import json
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main_model_common import TC, normalize_bars, tag_based_recap_mask
from main_model_cryptobert import CryptoBertMainModel

torch.set_num_threads(2)
all_news = pd.read_parquet(ROOT / "data/news/processed_news.parquet")
rows = []
for asset in ("BTC", "ETH"):
    bars = pd.read_pickle(f"/tmp/crypto-paper-{asset.lower()}-bars.pkl")
    news = all_news[all_news.asset == asset].reset_index(drop=True)
    enc = np.load(ROOT / f"data/processed/{asset.lower()}_news_embeddings_cryptobert.npz")
    np.testing.assert_array_equal(enc["timestamps"], news.unix_timestamp.to_numpy())
    order = np.argsort(enc["timestamps"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        order = order[(~tag_based_recap_mask(news))[order]]
    times = enc["timestamps"][order]
    feats = torch.from_numpy(enc["embeddings"][order].astype(np.float32))
    base_dir = ROOT / f"results/main/{asset.lower()}_cryptobert"
    for w in range(1, 7):
        meta = np.load(base_dir / f"metadata/window_{w}_meta.npz")
        normalized, _, _ = normalize_bars(bars, meta["train_idx"])
        indices = meta["test_idx"]
        bar_times = bars.timestamp.iloc[indices].to_numpy()
        left = np.searchsorted(times, bar_times-43200, side="left")
        right = np.searchsorted(times, bar_times, side="left")
        width = max(int(np.max(right-left)), 1)
        news_ix = left[:,None] + np.arange(width)[None,:]
        mask = news_ix < right[:,None]
        news_ix = np.where(mask, news_ix, 0)
        age = np.maximum((bar_times[:,None]-times[news_ix])/60, 0).astype(np.float32)
        price = torch.from_numpy(normalized[TC].iloc[indices].to_numpy(np.float32))
        for horizon in ("1h", "4h"):
            model = CryptoBertMainModel(768).eval()
            state = torch.load(base_dir / f"checkpoints/main_model_w{w}_{horizon}.pt",map_location="cpu",weights_only=True)
            model.load_state_dict(state)
            with torch.no_grad():
                raw = model.news_head(model.input_norm(feats))
                magnitude = F.softplus(raw[:,0]).numpy()
                rate = (.001+.499*torch.sigmoid(raw[:,1])).numpy()
                signal = (magnitude[news_ix]*np.exp(-rate[news_ix]*age)*mask).sum(1).astype(np.float32)
                correction = model.residual_head(torch.from_numpy(signal[:,None])).clamp(-2,2)
                pred = (model.forward_ts_only(price)*torch.exp(correction)).numpy().ravel()
            saved = np.load(base_dir / f"predictions/window_{w}_{horizon}_predictions.npz")
            np.testing.assert_allclose(pred,saved["main_model"],rtol=5e-5,atol=1e-10)
            error = float(np.max(np.abs(pred/saved["main_model"]-1)))
            rows.append(dict(asset=asset,window=w,horizon=horizon,N=len(pred),max_relative_error=error))
            print(f"Verified full NED forecasts: {asset} W{w} {horizon}, N={len(pred)}, relative error={error:.3g}",flush=True)

out = ROOT / "paper/generated"
pd.DataFrame(rows).to_csv(out / "main_forecast_checks.csv",index=False)
summary = dict(verified_models=len(rows),observations=sum(r["N"] for r in rows),
    max_relative_error=max(r["max_relative_error"] for r in rows),passed=True)
(out / "main_forecast_validation.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary))
