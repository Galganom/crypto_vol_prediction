"""Rebuild current price features and check archived targets/model predictions.

Slower optional verification; no fitting or input mutation. Run from repo root:
  .venv/bin/python paper/verify_price_pipeline.py
Caches rebuilt frames only in /tmp for optional follow-up diagnostics.
"""
from pathlib import Path
import json
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import numpy as np
import pandas as pd
import torch
from main_model_common import build_price_bars, align_bars_with_news, normalize_bars, TC, WINDOWS
from main_model_cryptobert import CryptoBertMainModel

torch.set_num_threads(2)
rows = []
for asset in ("BTC", "ETH"):
    print(f"Rebuilding {asset} price features with build_price_bars ...", flush=True)
    bars = build_price_bars(asset)
    times = np.load(f"data/processed/{asset.lower()}_news_embeddings_cryptobert.npz")["timestamps"]
    bars, _, _ = align_bars_with_news(bars, np.sort(times))
    bars.to_pickle(f"/tmp/crypto-paper-{asset.lower()}-bars.pkl")
    print(f"{asset}: {len(bars)} bars; {bars.datetime.iloc[0]} -- {bars.datetime.iloc[-1]}", flush=True)
    base = ROOT / f"results/main/{asset.lower()}_cryptobert"
    for w in range(1, 7):
        meta = np.load(base / f"metadata/window_{w}_meta.npz")
        vs, ve, ts, te = (pd.Timestamp(x, tz="UTC") for x in WINDOWS[w + 1])
        for name, mask in (("train_idx", bars.datetime < vs),
                           ("val_idx", (bars.datetime >= vs) & (bars.datetime <= ve)),
                           ("test_idx", (bars.datetime >= ts) & (bars.datetime <= te))):
            np.testing.assert_array_equal(np.flatnonzero(mask), meta[name])
        normalized, mean, std = normalize_bars(bars, meta["train_idx"])
        np.testing.assert_allclose(mean, meta["norm_mean"], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(std, meta["norm_std"], rtol=1e-6, atol=1e-8)
        for h in ("1h", "4h"):
            data = np.load(base / f"predictions/window_{w}_{h}_predictions.npz", allow_pickle=True)
            idx = meta["test_idx"]
            np.testing.assert_array_equal(bars.datetime.iloc[idx].astype(str), data["datetime"])
            np.testing.assert_allclose(bars[f"RV_{h}"].iloc[idx], data["actual"], rtol=1e-10, atol=1e-15)
            model = CryptoBertMainModel(768).eval()
            model.load_state_dict(torch.load(base / f"checkpoints/main_model_w{w}_{h}.pt", map_location="cpu", weights_only=True))
            with torch.no_grad():
                pred = model.forward_ts_only(torch.tensor(normalized[TC].iloc[idx].to_numpy(np.float32))).numpy().ravel()
                valid = model.forward_ts_only(torch.tensor(normalized[TC].iloc[meta["val_idx"]].to_numpy(np.float32))).ravel()
                actual = torch.tensor(bars[f"RV_{h}"].iloc[meta["val_idx"]].to_numpy(np.float32))
                loss = torch.nn.functional.huber_loss(torch.log(valid + 1e-10), torch.log(actual + 1e-10), delta=.5).item()
            np.testing.assert_allclose(pred, data["ts_mlp"], rtol=5e-5, atol=1e-9)
            history = pd.read_csv(base / f"loss_curves/main_model_w{w}_{h}_loss.csv")
            recorded = history[history.phase == "P1"].val_loss.min()
            assert abs(loss - recorded) < 1e-5, (asset, w, h, loss, recorded)
            rows.append(dict(asset=asset, window=w, horizon=h, bars=len(bars),
                first_bar=str(bars.datetime.iloc[0]), last_bar=str(bars.datetime.iloc[-1]),
                target_max_error=float(np.max(np.abs(bars[f"RV_{h}"].iloc[idx].to_numpy()-data["actual"]))),
                ts_max_relative_error=float(np.max(np.abs(pred/data["ts_mlp"]-1))),
                p1_huber_val=loss, recorded_p1_val=recorded))
            print(f"{asset} W{w} {h}: targets, normalization, TS predictions and Huber validation verified", flush=True)
out = ROOT / "paper/generated"
out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out / "price_pipeline_checks.csv", index=False)
print(json.dumps({"checked_runs": len(rows), "status": "passed"}), flush=True)
