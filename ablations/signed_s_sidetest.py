"""Side test — does allowing the per-article `s` value to be SIGNED improve
the BTC 4h Main Model?

Tests three variants of the news head's first output channel on a single
expanding window (BTC W3, source W5: test 2024-01-01 → 2024-06-30):

    softplus  (baseline, current behaviour — s in [0, ∞))
    tanh      (bounded signed,  s in [-1, 1])
    raw       (unbounded signed, s any real)

Everything else is identical to main_model_cryptobert.CryptoBertMainModel:
  - full 768-dim CryptoBERT input, LayerNorm
  - news_head = Linear(768, 128) → ReLU → Linear(128, 32) → ReLU → Linear(32, 2)
  - λ = 0.001 + 0.499·sigmoid(out[:, :, 1])
  - signal = Σ s_i · exp(-λ_i · dt_i)
  - output = base · exp(residual_head(signal).clamp(-2, 2))

Shared P1 state is reused so the only thing varying is the news head's
sign behaviour. Reports val + test R² per variant.
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from main_model_common import (  # noqa: E402
    NT, P1_SEED, SEED, TC, WINDOWS, AUX_WEIGHT, LOG_EPS,
    align_bars_with_news, build_price_bars, cfg, clone_ts_state,
    compute_metrics, level_loss, load_news_features, load_ts_state, log,
    make_split_loaders, normalize_bars, set_seed,
)
from main_model_cryptobert import (  # noqa: E402
    CB_P1_LR, CB_P2_LR, CB_BATCH_SIZE, CB_GRAD_CLIP, CB_HIDDEN,
    CB_P1_MAX_EPOCHS, CB_P2_MAX_EPOCHS, CB_P1_PATIENCE, CB_P2_PATIENCE,
    load_cryptobert_embeddings, make_news_head,
)

TARGET_COL = "RV_4h"
WINDOW_NUM = 3       # BTC W3 (test 2024-01-01 → 2024-06-30)
OUT_DIR = REPO_ROOT / "results/sidetest_signed_s_btc_w3_4h"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CryptoBertMainModel(nn.Module):
    """Same as main_model_cryptobert.CryptoBertMainModel but the s-channel
    transformation is configurable: 'softplus' (baseline), 'tanh', or 'raw'."""

    def __init__(self, feat_dim, hidden=CB_HIDDEN, s_mode="softplus"):
        super().__init__()
        assert s_mode in {"softplus", "tanh", "raw"}, s_mode
        self.s_mode = s_mode
        self.ts_net = nn.Sequential(
            nn.Linear(NT, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.ts_head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Softplus(),
        )
        self.input_norm = nn.LayerNorm(feat_dim)
        self.news_head = make_news_head(feat_dim, hidden)
        self.residual_head = nn.Linear(1, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.aux_head = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Softplus(),
        )

    def _s(self, raw):
        if self.s_mode == "softplus":
            return F.softplus(raw)
        if self.s_mode == "tanh":
            return torch.tanh(raw)
        return raw  # "raw" — unbounded signed

    def _event_signal(self, nf, nt, bt, mk):
        bsz, n_news, feat_dim = nf.shape
        x = self.input_norm(nf.reshape(bsz * n_news, feat_dim))
        out = self.news_head(x).reshape(bsz, n_news, 2)
        s = self._s(out[:, :, 0:1])
        lam = 0.001 + 0.499 * torch.sigmoid(out[:, :, 1:2])
        dt = (bt.unsqueeze(1) - nt).float().clamp(min=0) / 60.0
        return (s * torch.exp(-lam * dt.unsqueeze(2)) * mk.unsqueeze(2).float()).sum(1)

    def forward_ts_only(self, tf):
        return self.ts_head(self.ts_net(tf))

    def forward(self, tf, nf, nt, bt, mk, tm=None):
        base = self.forward_ts_only(tf)
        signal = self._event_signal(nf, nt, bt, mk)
        return base * torch.exp(self.residual_head(signal).clamp(-2, 2))


def train_p1(loaders, feat_dim, device):
    set_seed(P1_SEED)
    model = CryptoBertMainModel(feat_dim).to(device)
    for n, p in model.named_parameters():
        if n.startswith(("ts_net", "ts_head")):
            p.requires_grad = True
        else:
            p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CB_P1_LR)
    best, pat, best_sd = float("inf"), 0, None
    for ep in range(CB_P1_MAX_EPOCHS):
        model.train()
        for batch in loaders["train"]:
            tf = batch["tf"].to(device); tgt = batch["tg"].to(device)
            loss = level_loss(model.forward_ts_only(tf), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in loaders["val"]:
                tf = batch["tf"].to(device); tgt = batch["tg"].to(device)
                vl += level_loss(model.forward_ts_only(tf), tgt).item() * tf.size(0)
                vn += tf.size(0)
        vl /= max(vn, 1)
        if vl < best:
            best, pat, best_sd = vl, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if pat >= CB_P1_PATIENCE:
            break
    model.load_state_dict(best_sd)
    return clone_ts_state(model), float(best)


def train_p2(loaders, feat_dim, p1_state, device, s_mode):
    set_seed(SEED)
    model = CryptoBertMainModel(feat_dim, s_mode=s_mode).to(device)
    load_ts_state(model, p1_state)
    # Freeze TS
    for n, p in model.named_parameters():
        if n.startswith(("ts_net", "ts_head")):
            p.requires_grad = False
        else:
            p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CB_P2_LR)
    best, pat, best_sd = float("inf"), 0, None
    for ep in range(CB_P2_MAX_EPOCHS):
        model.train()
        for batch in loaders["train"]:
            tf = batch["tf"].to(device); nf = batch["nf"].to(device)
            nt = batch["nt"].to(device); bt = batch["bt"].to(device)
            mk = batch["mk"].to(device); tgt = batch["tg"].to(device)
            loss = level_loss(model(tf, nf, nt, bt, mk), tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CB_GRAD_CLIP)
            opt.step()
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in loaders["val"]:
                tf = batch["tf"].to(device); nf = batch["nf"].to(device)
                nt = batch["nt"].to(device); bt = batch["bt"].to(device)
                mk = batch["mk"].to(device); tgt = batch["tg"].to(device)
                vl += level_loss(model(tf, nf, nt, bt, mk), tgt).item() * tf.size(0)
                vn += tf.size(0)
        vl /= max(vn, 1)
        if vl < best:
            best, pat, best_sd = vl, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        log(f"  [{s_mode}] P2 ep {ep+1:3d} val={vl:.4f} pat={pat}{'  *' if pat == 0 else ''}")
        if pat >= CB_P2_PATIENCE:
            break
    model.load_state_dict(best_sd)
    return model, float(best)


def predict_main_and_ts(model, p1_state, loader, feat_dim, device):
    ts_model = CryptoBertMainModel(feat_dim).to(device)
    load_ts_state(ts_model, p1_state)
    model.eval(); ts_model.eval()
    ts_preds, main_preds = [], []
    with torch.no_grad():
        for batch in loader:
            tf = batch["tf"].to(device); nf = batch["nf"].to(device)
            nt = batch["nt"].to(device); bt = batch["bt"].to(device)
            mk = batch["mk"].to(device)
            ts_preds.append(ts_model.forward_ts_only(tf).cpu().numpy())
            main_preds.append(model(tf, nf, nt, bt, mk).cpu().numpy())
    return np.concatenate(ts_preds).squeeze(1), np.concatenate(main_preds).squeeze(1)


def main():
    log(f"Device: {DEVICE}")
    log(f"Window: BTC W{WINDOW_NUM} (source W{WINDOW_NUM+2}), horizon 4h")

    bars = build_price_bars("BTC")
    news = load_news_features("BTC")
    bars, left, right = align_bars_with_news(bars, news["timestamps"])
    cb_emb = load_cryptobert_embeddings("BTC", news["timestamps"])
    log(f"Bars: {len(bars):,}; CryptoBERT 768-dim, {cb_emb.shape}")

    val_start, val_end, test_start, test_end = WINDOWS[WINDOW_NUM + 1]  # source window offset = 2
    vs = pd.Timestamp(val_start, tz="UTC"); ve = pd.Timestamp(val_end, tz="UTC")
    ts = pd.Timestamp(test_start, tz="UTC"); te = pd.Timestamp(test_end, tz="UTC")
    bars_dt = bars["datetime"]
    train_idx = np.where(bars_dt < vs)[0]
    val_idx = np.where((bars_dt >= vs) & (bars_dt <= ve))[0]
    test_idx = np.where((bars_dt >= ts) & (bars_dt <= te))[0]
    log(f"Train={len(train_idx):,} Val={len(val_idx):,} Test={len(test_idx):,}")

    bars_norm, _, _ = normalize_bars(bars, train_idx)
    actual = bars[TARGET_COL].iloc[test_idx].to_numpy(dtype=np.float64)
    zero_topic = np.zeros((len(news["timestamps"]), 1), dtype=np.float32)
    loaders = make_split_loaders(
        bars_norm, TARGET_COL, train_idx, val_idx, test_idx, left, right,
        cb_emb, zero_topic, news, CB_BATCH_SIZE,
    )

    # Shared P1 — try to load from cached checkpoint, else train
    p1_ckpt = REPO_ROOT / f"results/main/btc_cryptobert_v2/checkpoints/p1_model_w{WINDOW_NUM}_4h.pt"
    if p1_ckpt.exists():
        log(f"Loading cached P1 from {p1_ckpt}")
        model = CryptoBertMainModel(cb_emb.shape[1]).to(DEVICE)
        model.load_state_dict(torch.load(p1_ckpt, map_location=DEVICE, weights_only=True))
        p1_state = clone_ts_state(model)
        # quick eval on val for reference
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in loaders["val"]:
                tf = batch["tf"].to(DEVICE); tgt = batch["tg"].to(DEVICE)
                vl += level_loss(model.forward_ts_only(tf), tgt).item() * tf.size(0)
                vn += tf.size(0)
        p1_val = vl / max(vn, 1)
        log(f"  Cached P1 val: {p1_val:.4f}")
    else:
        log("Training P1...")
        t0 = time.time()
        p1_state, p1_val = train_p1(loaders, cb_emb.shape[1], DEVICE)
        log(f"  P1 val={p1_val:.4f} [{time.time()-t0:.0f}s]")

    rows = []
    for s_mode in ["softplus", "tanh", "raw"]:
        log("")
        log(f"=== Training P2 with s_mode={s_mode} ===")
        t0 = time.time()
        model, p2_val = train_p2(loaders, cb_emb.shape[1], p1_state, DEVICE, s_mode)
        ts_pred, main_pred = predict_main_and_ts(model, p1_state, loaders["test"], cb_emb.shape[1], DEVICE)
        ts_m = compute_metrics(ts_pred, actual)
        main_m = compute_metrics(main_pred, actual)
        elapsed = time.time() - t0
        delta_r2 = main_m["R2"] - ts_m["R2"]
        log(f"  s_mode={s_mode}: P2 val={p2_val:.4f}  "
            f"Main R2={main_m['R2']:.4f}  TS R2={ts_m['R2']:.4f}  Δ={delta_r2:+.4f}  "
            f"[{elapsed:.0f}s]")
        rows.append({
            "s_mode": s_mode, "p1_val": p1_val, "p2_val": p2_val,
            "ts_r2": ts_m["R2"], "main_r2": main_m["R2"], "delta_r2": delta_r2,
            "ts_mse": ts_m["MSE"], "main_mse": main_m["MSE"],
            "ts_qlike": ts_m["QLIKE"], "main_qlike": main_m["QLIKE"],
            "ts_da": ts_m["DA"], "main_da": main_m["DA"],
            "elapsed_s": elapsed,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "summary.csv", index=False, float_format="%.6f")
    log("\n=== Summary ===")
    log(df[["s_mode", "p2_val", "main_r2", "delta_r2", "main_mse", "main_qlike"]].to_string(index=False))
    log(f"\nSaved: {OUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
