"""Audit new experiment artifacts and independently reconstruct CPU forecasts.

No fitting and no CUDA required. Use --require-complete for the final audit.
The direct forecast calculation below does not call the training runner's
Variant, correction, kernel, static_features, or predict implementations.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main_model_common import TC, normalize_bars, compute_metrics
from paper.run_missing_experiments import load_inputs, VARIANTS

OUT = ROOT / "results/ablations/paper_revision"
GEN = ROOT / "paper/generated"
torch.set_num_threads(2)


def linear(x, state, name):
    return F.linear(x, state[name + ".weight"], state[name + ".bias"])


def direct_prediction(state, variant, dest, bar_times, x, times, emb, sent):
    base = F.softplus(linear(F.relu(linear(F.relu(linear(F.relu(
        linear(torch.from_numpy(x), state, "ts_net.0")), state, "ts_net.3")),
        state, "ts_head.0")), state, "ts_head.2")).numpy().ravel()
    if variant == "pca256_decay":
        pca = np.load(dest / "pca.npz")
        article_features = (emb @ pca["components"].T - pca["mean"] @ pca["components"].T).astype(np.float32)
    else:
        article_features = sent if variant == "sentiment_decay" else emb
    hours = 6 if variant == "lookback_6h" else 24 if variant == "lookback_24h" else 12
    corrections = []
    for timestamp in bar_times:
        begin = timestamp // 86400 * 86400 if variant == "daily_sentiment" else timestamp - hours * 3600
        lo, hi = np.searchsorted(times, [begin, timestamp], side="left")
        ages = ((timestamp - times[lo:hi]) / 60).astype(np.float32)
        if variant in {"news_count", "sentiment_recency", "daily_sentiment", "embedding_mean"}:
            if variant == "news_count":
                features = np.array([np.log1p(hi-lo)], dtype=np.float32)
                correction = linear(torch.from_numpy(features), state, "static_head")
            else:
                vectors = emb[lo:hi] if variant == "embedding_mean" else sent[lo:hi]
                weights = np.ones(hi-lo, dtype=np.float32)
                if variant == "sentiment_recency": weights /= 1 + ages
                features = (vectors * weights[:, None]).sum(0) / max(weights.sum(), 1e-8)
                norm = np.load(dest / "feature_norm.npz")
                features = ((features - norm["mean"]) / norm["std"]).astype(np.float32)
                correction = linear(F.relu(linear(torch.from_numpy(features), state, "static_head.0")), state, "static_head.2")
            corrections.append(float(correction.item()))
            continue
        if hi == lo:
            signal = 0.
        else:
            features = torch.from_numpy(article_features[lo:hi])
            if "input_norm.weight" in state:
                features = F.layer_norm(features, (features.shape[1],), state["input_norm.weight"], state["input_norm.bias"])
            raw = linear(F.relu(linear(F.relu(linear(features, state, "news_head.0")), state, "news_head.2")), state, "news_head.4").numpy()
            magnitude = np.logaddexp(0, raw[:, 0])
            if variant == "constant":
                weight = np.ones_like(ages)
            elif variant == "power":
                weight = (1 + ages / 60) ** (-np.logaddexp(0, raw[:, 1]))
            elif variant == "linear":
                weight = np.maximum(0, 1 - ages / (1 + 60*np.logaddexp(0, raw[:, 1])))
            else:
                rate = .001 + .499 / (1 + np.exp(-raw[:, 1]))
                weight = np.exp(-rate * ages)
            signal = np.sum(magnitude * weight)
        corrections.append(float(state["residual_head.weight"].item()*signal + state["residual_head.bias"].item()))
    return base * np.exp(np.clip(np.asarray(corrections), -2, 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    bars, times, emb, sent = load_inputs()
    rows = []
    for w in range(1, 7):
        meta = np.load(ROOT / f"results/main/btc_cryptobert/metadata/window_{w}_meta.npz")
        old = np.load(ROOT / f"results/main/btc_cryptobert/predictions/window_{w}_4h_predictions.npz")
        p1_path = ROOT / f"results/main/btc_cryptobert/checkpoints/p1_model_w{w}_4h.pt"
        p1 = torch.load(p1_path, map_location="cpu", weights_only=True)
        normed, _, _ = normalize_bars(bars, meta["train_idx"])
        for variant in VARIANTS:
            dest = OUT / f"window_{w}/{variant}"
            if not (dest / "metrics.json").exists(): continue
            report = json.loads((dest / "metrics.json").read_text())
            data = np.load(dest / "predictions.npz")
            np.testing.assert_array_equal(data["actual"], old["actual"])
            np.testing.assert_array_equal(data["test_idx"], meta["test_idx"])
            np.testing.assert_array_equal(data["ts_mlp"], old["ts_mlp"])
            metrics = compute_metrics(data["prediction"], data["actual"])
            for key in ("R2", "DA", "MSE", "QLIKE"):
                assert abs(metrics[key] - report[key]) < 1e-8
            assert report["p1_sha256"] == hashlib.sha256(p1_path.read_bytes()).hexdigest()
            history = pd.read_csv(dest / "loss.csv")
            assert len(history) == report["epochs"] <= 120
            assert abs(history.validation_loss.min()-report["best_validation_loss"]) < 1e-10
            assert len(history) == 120 or history.patience.iloc[-1] == 20
            state = torch.load(dest / "checkpoint.pt", map_location="cpu", weights_only=True)
            for name, value in p1.items():
                if name.startswith(("ts_net", "ts_head")):
                    assert torch.equal(state[name], value), (w, variant, name)
            assert state["ts_scale"].item() == 1.
            if variant == "pca256_decay":
                pca = np.load(dest / "pca.npz")
                training_times = bars.timestamp.iloc[meta["train_idx"]]
                assert pca["train_article_timestamps"].min() >= training_times.iloc[0]
                assert pca["train_article_timestamps"].max() <= training_times.iloc[-1]
                assert pca["components"].shape == (256,768)
            take = np.unique(np.r_[np.linspace(0,len(data["prediction"])-1,96,dtype=int),
                np.argmin(data["prediction"]),np.argmax(data["prediction"])])
            ix = data["test_idx"][take]
            with torch.no_grad():
                calculated = direct_prediction(state, variant, dest,
                    bars.timestamp.iloc[ix].to_numpy(), normed[TC].iloc[ix].to_numpy(np.float32), times, emb, sent)
            np.testing.assert_allclose(calculated, data["prediction"][take], rtol=5e-5, atol=1e-10)
            error = float(np.max(np.abs(calculated / data["prediction"][take] - 1)))
            rows.append(dict(window=w,variant=variant,observations_checked=len(take),max_prediction_relative_error=error,
                frozen_ts=True,metrics_verified=True,validation_selection_verified=True))
            print(f"Verified W{w} {variant}: CPU forecast relative error {error:.3g}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(GEN / "new_experiment_checks.csv", index=False)
    if args.require_complete:
        assert len(frame) == 72 and frame.groupby("variant").window.nunique().eq(6).all()
    summary = dict(verified_models=len(frame),forecast_observations=int(frame.observations_checked.sum()),
        max_prediction_relative_error=float(frame.max_prediction_relative_error.max()),required_models=72,
        complete=len(frame)==72,independent_cpu_reconstruction=True)
    (GEN / "new_experiment_validation.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary))


if __name__ == "__main__": main()
