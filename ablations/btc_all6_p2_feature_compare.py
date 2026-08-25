"""BTC 4h all-window P2 feature comparison.

Feature representations:
  - sentiment_decay: sentiment probabilities, 3-dim
  - cryptobert_pca256_decay: CryptoBERT embeddings reduced by per-window PCA-256
  - cryptobert_raw_decay: original CryptoBERT embeddings, 768-dim
  - finbert_raw_decay: original FinBERT CLS embeddings, 768-dim

For every window and seed, a single P1 TS branch is trained and shared by all
feature variants. Each variant then trains P2 with the TS branch frozen.
"""
import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from main_model_common import (
    HORIZONS, TC, WINDOWS, align_bars_with_news, build_price_bars, cfg,
    clone_ts_state, compute_metrics, dm_test, get_horizon_cfg, level_loss,
    load_news_features, load_ts_state, log, make_split_loaders, normalize_bars,
    set_seed,
)
from main_model_cryptobert import (
    CB_AUX_WEIGHT, CB_BATCH_SIZE, CB_GRAD_CLIP, CB_HIDDEN, CB_LOSS_LOG_EVERY,
    CB_P1_LR, CB_P1_MAX_EPOCHS, CB_P1_PATIENCE, CB_P2_LR, CB_P2_MAX_EPOCHS,
    CB_P2_PATIENCE, CryptoBertMainModel, _log_loss_epoch,
    _record_epoch, _save_loss_history, fit_cryptobert_pca,
    load_cryptobert_embeddings, resolve_cryptobert_window_map,
)


DEFAULT_WINDOWS = [1, 2, 3, 4, 5, 6]
DEFAULT_SEEDS = [7, 42, 2024]
DEFAULT_PCA_K = 256
FINBERT_CLS_PATH = "data/processed/btc_news_embeddings_finbert_cls.npz"
MODEL_SPECS = OrderedDict([
    ("sentiment_decay", "sentiment"),
    ("cryptobert_pca256_decay", "cryptobert_pca256"),
    ("cryptobert_raw_decay", "cryptobert_raw"),
    ("finbert_raw_decay", "finbert_raw"),
])


class TsDataset(Dataset):
    def __init__(self, df, target_col):
        self.tf = df[TC].values.astype(np.float32)
        self.tg = df[target_col].values.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.tf)

    def __getitem__(self, i):
        return {
            "tf": torch.tensor(self.tf[i], dtype=torch.float32),
            "tg": torch.tensor(self.tg[i], dtype=torch.float32),
        }


def _make_ts_loaders(bars_norm, target_col, train_idx, val_idx, test_idx, batch_size):
    return {
        "train": DataLoader(
            TsDataset(bars_norm.iloc[train_idx].reset_index(drop=True), target_col),
            batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
        ),
        "val": DataLoader(
            TsDataset(bars_norm.iloc[val_idx].reset_index(drop=True), target_col),
            batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
        ),
        "test": DataLoader(
            TsDataset(bars_norm.iloc[test_idx].reset_index(drop=True), target_col),
            batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
        ),
    }


def _eval_ts_loss(model, loader, device):
    model.eval()
    loss_sum, n_obs = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            tf = batch["tf"].to(device)
            tgt = batch["tg"].to(device)
            loss_sum += level_loss(model.forward_ts_only(tf), tgt).item() * tf.size(0)
            n_obs += tf.size(0)
    return loss_sum / max(n_obs, 1)


def _predict_ts(p1_state, loader, device):
    model = CryptoBertMainModel(1, hidden=CB_HIDDEN).to(device)
    load_ts_state(model, p1_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            tf = batch["tf"].to(device)
            preds.append(model.forward_ts_only(tf).cpu().numpy())
    return np.concatenate(preds).squeeze(1)


def _train_shared_p1(loaders, device, args, seed, window_num, horizon):
    set_seed(seed)
    model = CryptoBertMainModel(1, hidden=CB_HIDDEN).to(device)
    for p in model.news_head.parameters():
        p.requires_grad = False
    for p in model.residual_head.parameters():
        p.requires_grad = False
    for p in model.aux_head.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.p1_lr)

    history = []
    best, pat, best_sd = float("inf"), 0, None
    start = time.time()
    loss_csv = os.path.join(args.results_dir, "loss_curves", f"shared_p1_w{window_num}_s{seed}_{horizon}.csv")
    loss_png = os.path.join(args.results_dir, "loss_curves", f"shared_p1_w{window_num}_s{seed}_{horizon}.png")
    log(f"  Shared TS P1 seed={seed}")
    for ep in range(args.p1_max_epochs):
        model.train()
        train_sum, train_n = 0.0, 0
        for batch in loaders["train"]:
            tf = batch["tf"].to(device)
            tgt = batch["tg"].to(device)
            loss = level_loss(model.forward_ts_only(tf), tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_sum += loss.item() * tf.size(0)
            train_n += tf.size(0)
        train_loss = train_sum / max(train_n, 1)
        val_loss = _eval_ts_loss(model, loaders["val"], device)
        improved = val_loss < best
        if improved:
            best, pat = val_loss, 0
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        _record_epoch(
            history, "P1", ep + 1, ep + 1, train_loss, train_loss,
            val_loss, best, pat, None, args.p1_lr, improved, time.time() - start,
        )
        _log_loss_epoch(
            "P1", ep + 1, train_loss, train_loss, val_loss, best, pat,
            lr_ts=args.p1_lr, improved=improved, loss_log_every=args.loss_log_every,
        )
        _save_loss_history(history, loss_csv)
        if pat >= args.p1_patience:
            break

    if best_sd is None:
        raise RuntimeError("Shared P1 did not run.")
    model.load_state_dict(best_sd)
    _save_loss_history(history, loss_csv, loss_png)
    p1_state = clone_ts_state(model)
    torch.save(
        p1_state,
        os.path.join(args.results_dir, "checkpoints", f"shared_p1_ts_w{window_num}_s{seed}_{horizon}.pt"),
    )
    log(f"    Shared P1 best val: {best:.4f}")
    return p1_state, float(best)


def _load_aligned_embeddings(path, news_timestamps):
    data = np.load(path)
    si = np.argsort(data["timestamps"])
    ts = data["timestamps"][si]
    emb = data["embeddings"][si].astype(np.float32)
    if not np.array_equal(ts, news_timestamps):
        raise ValueError(f"Embedding timestamps do not align with news timestamps: {path}")
    return emb


def _eval_model_loss(model, loader, device):
    model.eval()
    loss_sum, n_obs = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            tf = batch["tf"].to(device)
            nf = batch["nf"].to(device)
            nt = batch["nt"].to(device)
            bt = batch["bt"].to(device)
            mk = batch["mk"].to(device)
            tgt = batch["tg"].to(device)
            loss_sum += level_loss(model(tf, nf, nt, bt, mk), tgt).item() * tf.size(0)
            n_obs += tf.size(0)
    return loss_sum / max(n_obs, 1)


def _predict_model(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            tf = batch["tf"].to(device)
            nf = batch["nf"].to(device)
            nt = batch["nt"].to(device)
            bt = batch["bt"].to(device)
            mk = batch["mk"].to(device)
            preds.append(model(tf, nf, nt, bt, mk).cpu().numpy())
    return np.concatenate(preds).squeeze(1)


def _train_p2_model(loaders, feat_dim, p1_state, device, args, seed, model_name, window_num, horizon):
    set_seed(seed)
    model = CryptoBertMainModel(feat_dim, hidden=CB_HIDDEN).to(device)
    load_ts_state(model, p1_state)
    history = []
    global_epoch = 0
    start = time.time()
    loss_csv = os.path.join(args.results_dir, "loss_curves", f"{model_name}_w{window_num}_s{seed}_{horizon}.csv")
    loss_png = os.path.join(args.results_dir, "loss_curves", f"{model_name}_w{window_num}_s{seed}_{horizon}.png")

    set_seed(seed)
    for p in model.parameters():
        p.requires_grad = True
    for p in model.ts_net.parameters():
        p.requires_grad = False
    for p in model.ts_head.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.p2_lr)
    best, pat, p2_sd = float("inf"), 0, None
    for ep in range(args.p2_max_epochs):
        model.train()
        train_sum, train_n = 0.0, 0
        for batch in loaders["train"]:
            tf = batch["tf"].to(device)
            nf = batch["nf"].to(device)
            nt = batch["nt"].to(device)
            bt = batch["bt"].to(device)
            mk = batch["mk"].to(device)
            tgt = batch["tg"].to(device)
            loss = level_loss(model(tf, nf, nt, bt, mk), tgt)
            if CB_AUX_WEIGHT > 0:
                loss = loss + CB_AUX_WEIGHT * level_loss(model.forward_aux(nf, nt, bt, mk), tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CB_GRAD_CLIP)
            opt.step()
            train_sum += loss.item() * tf.size(0)
            train_n += tf.size(0)
        train_loss = train_sum / max(train_n, 1)
        val_loss = _eval_model_loss(model, loaders["val"], device)
        improved = val_loss < best
        if improved:
            best, pat = val_loss, 0
            p2_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        global_epoch += 1
        _record_epoch(
            history, "P2", ep + 1, global_epoch, train_loss, train_loss,
            val_loss, best, pat, args.p2_lr, 0.0, improved, time.time() - start,
        )
        _log_loss_epoch(
            "P2", ep + 1, train_loss, train_loss, val_loss, best, pat,
            lr_news=args.p2_lr, lr_ts=0.0, improved=improved,
            loss_log_every=args.loss_log_every,
        )
        _save_loss_history(history, loss_csv)
        if pat >= args.p2_patience:
            break
    if p2_sd is None:
        raise RuntimeError(f"{model_name} P2 did not run.")
    p2_best = float(best)
    p2_epochs = global_epoch
    log(f"    P2 best val: {p2_best:.4f}")
    model.load_state_dict(p2_sd)
    _save_loss_history(history, loss_csv, loss_png)

    p2_pred = _predict_model(model, loaders["test"], device)
    torch.save(
        p2_sd,
        os.path.join(args.results_dir, "checkpoints", f"{model_name}_P2_w{window_num}_s{seed}_{horizon}.pt"),
    )
    return {"pred": p2_pred, "best_val": p2_best, "epochs": p2_epochs}


def _prediction_key(model_name):
    return model_name.replace("-", "_").replace(" ", "_")


def _metric_row(window_num, source_window_num, seed, test_period, horizon, model_name, actual, pred, meta):
    m = compute_metrics(pred, actual)
    return {
        "asset": "BTC",
        "horizon": horizon,
        "window": window_num,
        "source_window": source_window_num,
        "seed": seed,
        "test_period": test_period,
        "model": model_name,
        "MSE": m["MSE"],
        "QLIKE": m["QLIKE"],
        "R2": m["R2"],
        "DA": m["DA"],
        "N": len(actual),
        "best_val_loss": meta["best_val"],
        "epochs": meta["epochs"],
    }


def _save_metrics(rows, prediction_files, results_dir, horizon):
    df = pd.DataFrame(rows)
    metrics_path = os.path.join(results_dir, "p2_feature_metrics.csv")
    df.to_csv(metrics_path, index=False, float_format="%.10f")

    mean_df = (
        df.groupby(["asset", "horizon", "model"], as_index=False)
        .agg(
            MSE=("MSE", "mean"),
            QLIKE=("QLIKE", "mean"),
            R2=("R2", "mean"),
            DA=("DA", "mean"),
            best_val_loss=("best_val_loss", "mean"),
            epochs=("epochs", "mean"),
            N=("N", "sum"),
            n_windows=("window", "nunique"),
            n_seeds=("seed", "nunique"),
        )
    )
    mean_path = os.path.join(results_dir, "p2_feature_mean_metrics.csv")
    mean_df.to_csv(mean_path, index=False, float_format="%.10f")

    window_mean = (
        df.groupby(["asset", "horizon", "window", "model"], as_index=False)
        .agg(MSE=("MSE", "mean"), QLIKE=("QLIKE", "mean"), R2=("R2", "mean"), DA=("DA", "mean"))
    )
    window_mean_path = os.path.join(results_dir, "p2_feature_window_mean_metrics.csv")
    window_mean.to_csv(window_mean_path, index=False, float_format="%.10f")

    h_cfg = get_horizon_cfg(horizon)
    pair_rows = []
    pooled = {model: {"actual": [], "pred": []} for model in MODEL_SPECS}
    for path in prediction_files:
        data = np.load(path, allow_pickle=True)
        actual = data["actual"].astype(np.float64)
        for model in MODEL_SPECS:
            key = _prediction_key(model)
            if key in data:
                pooled[model]["actual"].append(actual)
                pooled[model]["pred"].append(data[key].astype(np.float64))
    for a, b in combinations(MODEL_SPECS.keys(), 2):
        if not pooled[a]["pred"] or not pooled[b]["pred"]:
            continue
        actual = np.concatenate(pooled[a]["actual"])
        ma = compute_metrics(np.concatenate(pooled[a]["pred"]), actual)
        mb = compute_metrics(np.concatenate(pooled[b]["pred"]), actual)
        t_mse, p_mse = dm_test(ma["mse_vec"], mb["mse_vec"], h_cfg["dm_bandwidth"])
        t_qlike, p_qlike = dm_test(ma["qlike_vec"], mb["qlike_vec"], h_cfg["dm_bandwidth"])
        pair_rows.append({
            "scope": "pooled_window_seed",
            "horizon": horizon,
            "model_a": a,
            "model_b": b,
            "MSE_a": ma["MSE"],
            "MSE_b": mb["MSE"],
            "QLIKE_a": ma["QLIKE"],
            "QLIKE_b": mb["QLIKE"],
            "DM_MSE_t": t_mse,
            "DM_MSE_p": p_mse,
            "DM_QLIKE_t": t_qlike,
            "DM_QLIKE_p": p_qlike,
            "N": len(actual),
        })
    pair_path = os.path.join(results_dir, "p2_feature_pairwise_dm.csv")
    pd.DataFrame(pair_rows).to_csv(pair_path, index=False, float_format="%.10f")

    log(f"Saved: {metrics_path}")
    log(f"Saved: {mean_path}")
    log(f"Saved: {window_mean_path}")
    log(f"Saved: {pair_path}")
    log("")
    log("Mean metrics:")
    log(mean_df.sort_values(["model"]).to_string(index=False))


def run(args):
    cfg("BTC")
    os.makedirs(args.results_dir, exist_ok=True)
    for sub in ["checkpoints", "loss_curves", "metadata", "predictions"]:
        os.makedirs(os.path.join(args.results_dir, sub), exist_ok=True)

    selected = OrderedDict((k, v) for k, v in MODEL_SPECS.items() if k in args.models)
    if not selected:
        raise ValueError("No models selected.")

    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizon = args.horizon
    target_col = get_horizon_cfg(horizon)["target"]
    window_map = resolve_cryptobert_window_map(args.windows)
    split_specs = [
        {
            "window_num": label,
            "source_window_num": source,
            "val_start": WINDOWS[source - 1][0],
            "val_end": WINDOWS[source - 1][1],
            "test_start": WINDOWS[source - 1][2],
            "test_end": WINDOWS[source - 1][3],
        }
        for source, label in sorted(window_map.items(), key=lambda item: item[1])
    ]

    log(f"Device: {device}")
    log(f"Results dir: {args.results_dir}")
    log(f"Horizon: {horizon}")
    log(f"Windows: {[s['window_num'] for s in split_specs]}")
    log(f"Seeds: {args.seeds}")
    log(f"Models: {list(selected.keys())}")
    log(f"LRs: P1={args.p1_lr:.2e}, P2={args.p2_lr:.2e}")

    bars = build_price_bars("BTC")
    news = load_news_features("BTC", rebuild_topics=False)
    bars, left, right = align_bars_with_news(bars, news["timestamps"])
    crypto_raw = load_cryptobert_embeddings("BTC", news["timestamps"])
    finbert_raw = (
        _load_aligned_embeddings(FINBERT_CLS_PATH, news["timestamps"])
        if "finbert_raw_decay" in selected else None
    )
    zero_topic = np.zeros((len(news["timestamps"]), 1), dtype=np.float32)
    bars_dt = bars["datetime"]

    rows = []
    prediction_files = []
    for split in split_specs:
        window_num = split["window_num"]
        source_window_num = split["source_window_num"]
        val_start, val_end = split["val_start"], split["val_end"]
        test_start, test_end = split["test_start"], split["test_end"]
        vs = pd.Timestamp(val_start, tz="UTC")
        ve = pd.Timestamp(val_end, tz="UTC")
        ts = pd.Timestamp(test_start, tz="UTC")
        te = pd.Timestamp(test_end, tz="UTC")
        train_idx = np.where(bars_dt < vs)[0]
        val_idx = np.where((bars_dt >= vs) & (bars_dt <= ve))[0]
        test_idx = np.where((bars_dt >= ts) & (bars_dt <= te))[0]
        if len(train_idx) < 1000 or len(val_idx) < 100 or len(test_idx) < 100:
            log(f"Window {window_num}: skipped")
            continue

        log("")
        log("=" * 76)
        log(
            f"Window {window_num} (source {source_window_num}): "
            f"test {test_start} -> {test_end}"
        )
        log(f"Train={len(train_idx):,} Val={len(val_idx):,} Test={len(test_idx):,}")
        log("=" * 76)

        bars_norm, norm_mean, norm_std = normalize_bars(bars, train_idx)
        train_news_mask = (
            (news["timestamps"] >= bars["timestamp"].iloc[train_idx[0]]) &
            (news["timestamps"] <= bars["timestamp"].iloc[train_idx[-1]])
        )
        crypto_pca = None
        pca = None
        if "cryptobert_pca256_decay" in selected:
            crypto_pca, pca = fit_cryptobert_pca(
                crypto_raw, train_news_mask, k=args.pca_k, seed=args.pca_seed,
            )
            log(
                f"  CryptoBERT PCA train articles={int(train_news_mask.sum()):,}, "
                f"k={args.pca_k}, explained={pca.explained_variance_ratio_.sum():.3f}"
            )

        meta_payload = {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "source_window": source_window_num,
        }
        if pca is not None:
            meta_payload.update({
                "pca_components": pca.components_,
                "pca_mean": pca.mean_,
                "pca_explained_variance_ratio": pca.explained_variance_ratio_,
            })
        np.savez(os.path.join(args.results_dir, "metadata", f"window_{window_num}_meta.npz"), **meta_payload)

        feature_map = {}
        if "sentiment_decay" in selected:
            feature_map["sentiment_decay"] = news["sentiment_probs"].astype(np.float32)
        if "cryptobert_pca256_decay" in selected:
            feature_map["cryptobert_pca256_decay"] = crypto_pca.astype(np.float32)
        if "cryptobert_raw_decay" in selected:
            feature_map["cryptobert_raw_decay"] = crypto_raw.astype(np.float32)
        if "finbert_raw_decay" in selected:
            feature_map["finbert_raw_decay"] = finbert_raw.astype(np.float32)

        actual = bars[target_col].iloc[test_idx].to_numpy(dtype=np.float64)
        test_period = f"{test_start}->{test_end}"
        ts_loaders = _make_ts_loaders(bars_norm, target_col, train_idx, val_idx, test_idx, args.batch_size)

        for seed in args.seeds:
            log("")
            log("-" * 76)
            log(f"Window {window_num} seed {seed}")
            log("-" * 76)
            p1_state, p1_best = _train_shared_p1(ts_loaders, device, args, seed, window_num, horizon)
            ts_pred = _predict_ts(p1_state, ts_loaders["test"], device)
            ts_metrics = compute_metrics(ts_pred, actual)
            log(
                f"    shared_TS_only                  R2={ts_metrics['R2']:.4f} "
                f"MSE={ts_metrics['MSE']:.5f} QLIKE={ts_metrics['QLIKE']:.5f} DA={ts_metrics['DA']:.3f}"
            )

            pred_payload = {
                "actual": actual,
                "datetime": bars["datetime"].iloc[test_idx].astype(str).values,
                "test_idx": test_idx,
                "window": window_num,
                "source_window": source_window_num,
                "seed": seed,
                "horizon": horizon,
                "target_col": target_col,
                "shared_p1_best_val": p1_best,
                "shared_TS_only": ts_pred,
            }

            for model_name, feat in feature_map.items():
                log("")
                log(f"  Feature model {model_name}: dim={feat.shape[1]}")
                loaders = make_split_loaders(
                    bars_norm, target_col, train_idx, val_idx, test_idx,
                    left, right, feat, zero_topic, news, args.batch_size,
                )
                t0 = time.time()
                payload = _train_p2_model(
                    loaders, feat.shape[1], p1_state, device, args, seed,
                    model_name, window_num, horizon,
                )
                pred = payload["pred"]
                pred_payload[_prediction_key(model_name)] = pred
                rows.append(_metric_row(
                    window_num, source_window_num, seed, test_period, horizon,
                    model_name, actual, pred, payload,
                ))
                m = compute_metrics(pred, actual)
                log(
                    f"    P2 {model_name:<30s} R2={m['R2']:.4f} "
                    f"MSE={m['MSE']:.5f} QLIKE={m['QLIKE']:.5f} DA={m['DA']:.3f} "
                    f"best_val={payload['best_val']:.4f} epochs={payload['epochs']}"
                )
                log(f"    {model_name} total time={time.time() - t0:.0f}s")

            pred_path = os.path.join(
                args.results_dir, "predictions",
                f"window_{window_num}_seed_{seed}_{horizon}_predictions.npz",
            )
            np.savez(pred_path, **pred_payload)
            prediction_files.append(pred_path)
            _save_metrics(rows, prediction_files, args.results_dir, horizon)

    with open(os.path.join(args.results_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "asset": "BTC",
            "horizon": horizon,
            "windows": args.windows,
            "source_windows": [s["source_window_num"] for s in split_specs],
            "seeds": args.seeds,
            "models": list(selected.keys()),
            "training": "P1_shared_TS_then_P2_frozen_TS",
            "cryptobert_raw_source": "data/processed/btc_news_embeddings_cryptobert.npz",
            "finbert_raw_source": FINBERT_CLS_PATH,
            "pca_k": args.pca_k,
            "batch_size": args.batch_size,
            "hidden": list(CB_HIDDEN) if not isinstance(CB_HIDDEN, int) else CB_HIDDEN,
            "lr_p1": args.p1_lr,
            "lr_p2": args.p2_lr,
            "p1_patience": args.p1_patience,
            "p2_patience": args.p2_patience,
            "aux_weight": CB_AUX_WEIGHT,
            "grad_clip": CB_GRAD_CLIP,
        }, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/ablations/btc_all6_p2_feature_compare")
    parser.add_argument("--horizon", choices=list(HORIZONS.keys()), default="4h")
    parser.add_argument("--windows", nargs="+", type=int, default=DEFAULT_WINDOWS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SPECS.keys()), default=list(MODEL_SPECS.keys()))
    parser.add_argument("--batch-size", type=int, default=CB_BATCH_SIZE)
    parser.add_argument("--pca-k", type=int, default=DEFAULT_PCA_K)
    parser.add_argument("--pca-seed", type=int, default=42)
    parser.add_argument("--p1-lr", type=float, default=CB_P1_LR)
    parser.add_argument("--p2-lr", type=float, default=CB_P2_LR)
    parser.add_argument("--p1-max-epochs", type=int, default=CB_P1_MAX_EPOCHS)
    parser.add_argument("--p2-max-epochs", type=int, default=CB_P2_MAX_EPOCHS)
    parser.add_argument("--p1-patience", type=int, default=CB_P1_PATIENCE)
    parser.add_argument("--p2-patience", type=int, default=CB_P2_PATIENCE)
    parser.add_argument("--loss-log-every", type=int, default=CB_LOSS_LOG_EVERY)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
