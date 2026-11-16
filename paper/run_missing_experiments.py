"""Matched BTC 4h P2 ablations for the restored paper.

Uses the archived P1 weights, all six original train/validation/test splits,
Huber(delta=.5), AdamW(5e-5), batch 64, 120 epochs/patience 20, frozen TS
weights (training-mode dropout retained). CUDA graphs remove Python launch
overhead; neither data subsampling nor reduced epoch caps are used by default.
Only a new results/ablations/paper_revision directory is written.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/crypto-paper-matplotlib")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from main_model_common import TC, build_price_bars, align_bars_with_news, normalize_bars, tag_based_recap_mask, compute_metrics
from main_model_cryptobert import CryptoBertMainModel, load_ts_state, set_seed

VARIANTS = ["constant", "power", "linear", "exponential", "news_count",
            "sentiment_recency", "daily_sentiment", "embedding_mean",
            "sentiment_decay", "pca256_decay", "lookback_6h", "lookback_24h"]
STATIC = {"news_count", "sentiment_recency", "daily_sentiment", "embedding_mean"}
MAIN = ROOT / "results/main/btc_cryptobert"


def load_inputs():
    path = Path("/tmp/crypto-paper-btc-bars.pkl")
    news = pd.read_parquet("data/news/processed_news.parquet")
    news = news[news.asset == "BTC"].reset_index(drop=True)
    emb = np.load("data/processed/btc_news_embeddings_cryptobert.npz")
    sent = np.load("data/processed/btc_news_sentiment.npz")
    assert np.array_equal(emb["timestamps"], news.unix_timestamp.to_numpy())
    if "timestamps" in sent.files:
        assert np.array_equal(sent["timestamps"], emb["timestamps"])
    order = np.argsort(emb["timestamps"])
    times = emb["timestamps"][order].astype(np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kept = (~tag_based_recap_mask(news))[order]
    bars = pd.read_pickle(path) if path.exists() else align_bars_with_news(build_price_bars("BTC"), times)[0]
    return bars, times[kept], emb["embeddings"][order][kept].astype(np.float32), sent["sentiment_probs"][order][kept].astype(np.float32)


def pairs(bar_times, times, hours=12, daily=False):
    low = bar_times - hours * 3600 if not daily else (bar_times // 86400) * 86400
    left = np.searchsorted(times, low, side="left")
    right = np.searchsorted(times, bar_times, side="left")
    count = right - left
    width = max(int(count.max()), 1)
    ids = left[:, None] + np.arange(width)[None, :]
    mask = ids < right[:, None]
    ids = np.where(mask, ids, 0).astype(np.int32)
    age = np.where(mask, (bar_times[:, None] - times[ids]) / 60., 0).astype(np.float32)
    return ids, age, mask


class Variant(CryptoBertMainModel):
    def __init__(self, dim, variant):
        super().__init__(dim)
        self.variant = variant
        if variant == "sentiment_decay":
            self.input_norm = nn.Identity()
        if variant in STATIC:
            if dim == 1:
                self.static_head = nn.Linear(1, 1)
                nn.init.zeros_(self.static_head.weight)
                nn.init.zeros_(self.static_head.bias)
            else:
                self.static_head = nn.Sequential(nn.Linear(dim, 16), nn.ReLU(), nn.Linear(16, 1))
                nn.init.zeros_(self.static_head[-1].weight)
                nn.init.zeros_(self.static_head[-1].bias)

    def kernel(self, alpha, age):
        if self.variant == "constant":
            return torch.ones_like(age)
        if self.variant == "power":
            return (1 + age / 60.).pow(-F.softplus(alpha))
        if self.variant == "linear":
            return F.relu(1 - age / (1 + 60 * F.softplus(alpha)))
        return torch.exp(-(.001 + .499 * torch.sigmoid(alpha)) * age)

    def correction(self, feat, ids, age, mask):
        if self.variant in STATIC:
            return self.static_head(feat).clamp(-2, 2)
        n, k = ids.shape
        output = self.news_head(self.input_norm(feat[ids].reshape(n * k, -1))).reshape(n, k, 2)
        signal = (F.softplus(output[..., 0]) * self.kernel(output[..., 1], age) * mask).sum(1, keepdim=True)
        return self.residual_head(signal).clamp(-2, 2)


def static_features(variant, pair, emb, sent):
    ids, age, mask = pair
    if variant == "news_count":
        return np.log1p(mask.sum(1, keepdims=True)).astype(np.float32)
    article_feat = emb if variant == "embedding_mean" else sent
    out = np.zeros((len(ids), article_feat.shape[1]), np.float32)
    for start in range(0, len(ids), 2048):
        stop = start + 2048
        weight = mask[start:stop].astype(np.float32)
        if variant == "sentiment_recency":
            weight /= 1 + age[start:stop]
        out[start:stop] = (article_feat[ids[start:stop]] * weight[..., None]).sum(1) / weight.sum(1, keepdims=True).clip(1e-8)
    return out


def predict(model, indices, tf, feat, ids, age, mask):
    model.eval()
    result = []
    with torch.no_grad():
        output = None if model.variant in STATIC else model.news_head(model.input_norm(feat))
        for start in range(0, len(indices), 4096):
            ix = indices[start:start+4096]
            base = model.forward_ts_only(tf[ix])
            if output is None:
                corr = model.static_head(feat[ix]).clamp(-2, 2)
            else:
                o = output[ids[ix]]
                signal = (F.softplus(o[..., 0]) * model.kernel(o[..., 1], age[ix]) * mask[ix]).sum(1, keepdim=True)
                corr = model.residual_head(signal).clamp(-2, 2)
            result.append((base * torch.exp(corr)).ravel())
    return torch.cat(result)


def train_one(args, window, variant, bars, times, emb, sent, pair_cache):
    dest = args.out / f"window_{window}/{variant}"
    if (dest / "metrics.json").exists():
        print(f"SKIP completed W{window} {variant}", flush=True)
        return
    dest.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    meta = np.load(MAIN / f"metadata/window_{window}_meta.npz")
    normed, mean, std = normalize_bars(bars, meta["train_idx"])
    np.testing.assert_allclose(mean, meta["norm_mean"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(std, meta["norm_std"], rtol=1e-6, atol=1e-8)
    archived = np.load(MAIN / f"predictions/window_{window}_4h_predictions.npz")
    np.testing.assert_array_equal(bars.RV_4h.iloc[meta["test_idx"]], archived["actual"])
    key = 6 if variant == "lookback_6h" else 24 if variant == "lookback_24h" else "daily" if variant == "daily_sentiment" else 12
    if key not in pair_cache:
        pair_cache[key] = pairs(bars.timestamp.to_numpy(np.int64), times, hours=key if key != "daily" else 24, daily=key == "daily")
    pair = pair_cache[key]
    features = emb
    pca = None
    if variant == "sentiment_decay":
        features = sent
    if variant == "pca256_decay":
        trtimes = bars.timestamp.iloc[meta["train_idx"]]
        train_news = (times >= trtimes.iloc[0]) & (times <= trtimes.iloc[-1])
        pca = PCA(n_components=256, random_state=42)
        pca.fit(emb[train_news])
        features = pca.transform(emb).astype(np.float32)
        np.savez(dest / "pca.npz", components=pca.components_, mean=pca.mean_, explained_variance_ratio=pca.explained_variance_ratio_, train_article_timestamps=times[train_news])
    if variant in STATIC:
        features = static_features(variant, pair, emb, sent)
        if variant != "news_count":
            fmean = features[meta["train_idx"]].mean(0)
            fstd = features[meta["train_idx"]].std(0) + 1e-8
            features = ((features - fmean) / fstd).astype(np.float32)
            np.savez(dest / "feature_norm.npz", mean=fmean, std=fstd)
    set_seed(7)
    model = Variant(features.shape[1], variant).cuda()
    p1_path = MAIN / f"checkpoints/p1_model_w{window}_4h.pt"
    p1 = torch.load(p1_path, map_location="cpu", weights_only=True)
    load_ts_state(model, {k: v for k, v in p1.items() if k.startswith(("ts_net", "ts_head"))})
    # Full-dimensional variants begin with precisely the archived untrained
    # article head and zero residual map from P1. Other input dimensions use
    # the same initialization seed and hidden widths.
    if variant not in STATIC and features.shape[1] == 768:
        model.load_state_dict(p1)
    for name, par in model.named_parameters():
        par.requires_grad = name.startswith(("input_norm", "news_head", "residual_head")) if variant not in STATIC else name.startswith("static_head")
    model.ts_scale.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    tf = torch.tensor(normed[TC].to_numpy(np.float32), device="cuda")
    target = torch.log(torch.tensor(bars.RV_4h.to_numpy(np.float32), device="cuda") + 1e-10)
    feat = torch.tensor(features, device="cuda")
    del features, normed
    ids = torch.tensor(pair[0], dtype=torch.long, device="cuda")
    age = torch.tensor(pair[1], device="cuda")
    mask = torch.tensor(pair[2], dtype=torch.float32, device="cuda")
    train_indices = meta["train_idx"]
    valid_indices = torch.tensor(meta["val_idx"], device="cuda")
    test_indices = torch.tensor(meta["test_idx"], device="cuda")
    model.eval()
    with torch.no_grad():
        tscheck = model.forward_ts_only(tf[test_indices]).cpu().numpy().ravel()
    np.testing.assert_allclose(tscheck, archived["ts_mlp"], rtol=5e-5, atol=1e-9)
    opt = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=.01, fused=True, capturable=True)
    batch_ix = torch.tensor(train_indices[:64], device="cuda")
    batch_weight = torch.ones(64, device="cuda")
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    model.train()

    def step():
        opt.zero_grad(set_to_none=True)
        base = model.forward_ts_only(tf[batch_ix])
        fc = feat[batch_ix] if variant in STATIC else feat
        correction = model.correction(fc, ids[batch_ix], age[batch_ix], mask[batch_ix])
        pred = (base * torch.exp(correction)).ravel()
        loss_vec = F.huber_loss(torch.log(pred + 1e-10), target[batch_ix], delta=.5, reduction="none")
        loss = (loss_vec * batch_weight).sum() / batch_weight.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, foreach=True)
        opt.step()
        return loss

    # Capture the complete fixed-shape optimizer step. Warm-up/capture updates
    # are discarded, including Adam moments and step counters.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_loss = step()
    model.load_state_dict(initial)
    for state in opt.state.values():
        for value in state.values():
            if torch.is_tensor(value): value.zero_()
    set_seed(42)
    generator = np.random.default_rng(42)
    best = float("inf")
    patience = 0
    history = []
    best_state = None
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        order = torch.tensor(generator.permutation(train_indices), device="cuda")
        batch_weight.fill_(1)
        for offset in range(0, len(order), 64):
            n = min(64, len(order) - offset)
            if n == 64:
                batch_ix.copy_(order[offset:offset+64])
            else:
                batch_ix[:n].copy_(order[offset:])
                batch_ix[n:].fill_(int(train_indices[0]))
                batch_weight[n:].zero_()
            graph.replay()
        validation = predict(model, valid_indices, tf, feat, ids, age, mask)
        vl = F.huber_loss(torch.log(validation + 1e-10), target[valid_indices], delta=.5).item()
        improved = vl < best
        if improved:
            best, patience = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        elapsed = time.monotonic() - start_time
        history.append(dict(epoch=epoch, validation_loss=vl, best_validation_loss=best, patience=patience, elapsed_s=elapsed))
        pd.DataFrame(history).to_csv(dest / "loss.csv", index=False)
        if epoch == 1 or epoch % 5 == 0 or patience >= args.patience:
            print(f"W{window} {variant} epoch {epoch} val={vl:.6f} best={best:.6f} patience={patience} elapsed={elapsed:.1f}s", flush=True)
        if patience >= args.patience: break
    model.load_state_dict(best_state)
    pred = predict(model, test_indices, tf, feat, ids, age, mask).cpu().numpy()
    assert np.isfinite(pred).all() and (pred > 0).all()
    metrics = compute_metrics(pred, archived["actual"])
    metrics = {k: metrics[k] for k in ("R2", "DA", "MSE", "QLIKE")}
    report = dict(window=window, variant=variant, asset="BTC", horizon="4h", **metrics,
        best_validation_loss=best, epochs=epoch, N=len(pred), elapsed_s=time.monotonic()-start_time,
        p1_sha256=hashlib.sha256(p1_path.read_bytes()).hexdigest())
    torch.save(best_state, dest / "checkpoint.pt")
    np.savez(dest / "predictions.npz", actual=archived["actual"], ts_mlp=archived["ts_mlp"], prediction=pred, test_idx=meta["test_idx"])
    (dest / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print("COMPLETED " + json.dumps(report), flush=True)
    del model, initial, best_state, opt, graph, tf, target, feat, ids, age, mask
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, nargs="+", default=list(range(1, 7)))
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--out", type=Path, default=ROOT/"results/ablations/paper_revision")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("A CUDA device is required for this runner")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    args.out.mkdir(parents=True, exist_ok=True)
    config = dict(asset="BTC", horizon="4h", windows=args.windows, variants=args.variants,
        batch_size=64, p2_lr=5e-5, weight_decay=.01, huber_delta=.5, max_epochs=args.max_epochs,
        patience=args.patience, initialization_seed=7, training_seed=42, ts_scale=1,
        dropout_in_frozen_ts=True, aux_weight=0, frozen_archived_p1=True,
        cuda_graph=True, optimizer="AdamW fused, capturable", device=torch.cuda.get_device_name(0),
        torch_version=torch.__version__, numpy_version=np.__version__,
        main_results_untouched=True, new_p2_runs=True)
    (args.out / "run_config.json").write_text(json.dumps(config, indent=2)+"\n")
    bars, times, emb, sent = load_inputs()
    cache = {}
    print(f"DATA bars={len(bars)}, retained articles={len(times)}, device={config['device']}", flush=True)
    for window in args.windows:
        for variant in args.variants:
            train_one(args, window, variant, bars, times, emb, sent, cache)
    reports=[json.loads(p.read_text()) for p in sorted(args.out.glob("window_*/*/metrics.json"))]
    frame=pd.DataFrame(reports)
    frame.to_csv(args.out/"metrics_per_window.csv",index=False)
    frame.groupby("variant").agg(R2=("R2","mean"), R2_std=("R2","std"), DA=("DA","mean"), MSE=("MSE","mean"), QLIKE=("QLIKE","mean"), n_windows=("window","nunique")).to_csv(args.out/"metrics_mean.csv")
    print("ALL REQUESTED ABLATIONS COMPLETED",flush=True)


if __name__ == "__main__": main()
