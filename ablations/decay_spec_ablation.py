"""Alternative decay specifications ablation — BTC 4h, all 8 windows.

Tests four ways to map the news head's two outputs (s, α) into a per-article
time-decayed contribution to the event signal:

  constant     contribution = s
                 (no temporal decay; α is ignored)

  exponential  contribution = s · exp(-λ · dt)
                 where λ = 0.001 + 0.499·sigmoid(α)  (the current setup)

  power        contribution = s / (1 + dt / T)^β
                 where β = softplus(α) and T = 60 min (fixed)
                 (heavy-tail Pareto-like decay; slower than exponential at long dt)

  linear       contribution = s · max(0, 1 - dt / T)
                 where T = softplus(α) * 60 + 1  (horizon in minutes, > 1)
                 (compact support — exactly zero after T minutes)

Everything else is identical to main_model_cryptobert.CryptoBertMainModel:
  - LayerNorm input + 768→128→32→2 news head
  - Shared P1 per window, loaded from
    results/main/btc_cryptobert_v2/checkpoints/p1_model_w{N}_4h.pt
    (trains fresh if missing)
  - Same P2 LR / batch / patience as production

Per-variant per-window R²/MSE/QLIKE/DA + pooled summary + DM tests vs the
exponential baseline.

Outputs: ablations/results/decay_spec_ablation_btc_4h/
"""
import argparse
import copy
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
    AUX_WEIGHT, LOG_EPS, NT, P1_SEED, SEED, TC, WINDOWS,
    align_bars_with_news, build_price_bars, cfg, clone_ts_state,
    compute_metrics, dm_test, huber_log_loss, level_loss, load_news_features,
    load_ts_state, log, make_split_loaders, normalize_bars, set_seed,
)
from main_model_cryptobert import (  # noqa: E402
    CB_BATCH_SIZE, CB_GRAD_CLIP, CB_HIDDEN, CB_HUBER_DELTA, CB_P1_LR,
    CB_P1_MAX_EPOCHS, CB_P1_PATIENCE, CB_P2_LR, CB_P2_MAX_EPOCHS, CB_P2_PATIENCE,
    load_cryptobert_embeddings, make_news_head,
)

TARGET_COL = "RV_4h"
HORIZON_LABEL = "4h"
OUT_DIR = REPO_ROOT / "results/ablations/decay_spec_btc_4h"
OUT_DIR.mkdir(parents=True, exist_ok=True)
P1_CKPT_DIR = REPO_ROOT / "results/main/btc_cryptobert_v2/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DECAY_SPECS = ["constant", "exponential", "power", "linear"]
POWER_T_MIN = 60.0      # fixed time scale for power-law decay
SOURCE_WINDOW_OFFSET = 2


class CryptoBertMainModelDecay(nn.Module):
    """News-head + 4 selectable per-article decay specifications."""

    def __init__(self, feat_dim, hidden=CB_HIDDEN, decay_spec="exponential"):
        super().__init__()
        assert decay_spec in DECAY_SPECS, decay_spec
        self.decay_spec = decay_spec
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

    def _decay_weight(self, alpha, dt):
        """Compute the decay weight given the news_head's 2nd output (alpha)
        and the per-pair article age dt (minutes). Returns same shape as dt."""
        if self.decay_spec == "constant":
            return torch.ones_like(dt)
        if self.decay_spec == "exponential":
            lam = 0.001 + 0.499 * torch.sigmoid(alpha)  # in (0.001, 0.500)
            return torch.exp(-lam * dt)
        if self.decay_spec == "power":
            beta = F.softplus(alpha)                    # > 0
            return 1.0 / (1.0 + dt / POWER_T_MIN).pow(beta)
        if self.decay_spec == "linear":
            T = F.softplus(alpha) * 60.0 + 1.0          # horizon in minutes, > 1
            return F.relu(1.0 - dt / T)

    def _event_signal(self, nf, nt, bt, mk):
        bsz, n_news, feat_dim = nf.shape
        x = self.input_norm(nf.reshape(bsz * n_news, feat_dim))
        out = self.news_head(x).reshape(bsz, n_news, 2)
        s = F.softplus(out[:, :, 0:1])
        alpha = out[:, :, 1:2]
        dt = (bt.unsqueeze(1) - nt).float().clamp(min=0).unsqueeze(2) / 60.0
        w = self._decay_weight(alpha, dt)
        return (s * w * mk.unsqueeze(2).float()).sum(1)

    def forward_ts_only(self, tf):
        return self.ts_head(self.ts_net(tf))

    def forward(self, tf, nf, nt, bt, mk, tm=None):
        base = self.forward_ts_only(tf)
        signal = self._event_signal(nf, nt, bt, mk)
        return base * torch.exp(self.residual_head(signal).clamp(-2, 2))


def train_p1(loaders, feat_dim, device):
    set_seed(P1_SEED)
    model = CryptoBertMainModelDecay(feat_dim).to(device)
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(("ts_net", "ts_head"))
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
    return clone_ts_state(model), float(best), best_sd


def load_or_train_p1(window_num, loaders, feat_dim, device):
    p1_path = P1_CKPT_DIR / f"p1_model_w{window_num}_{HORIZON_LABEL}.pt"
    if p1_path.exists():
        m = CryptoBertMainModelDecay(feat_dim).to(device)
        m.load_state_dict(torch.load(p1_path, map_location=device, weights_only=True))
        # quick val
        m.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in loaders["val"]:
                tf = batch["tf"].to(device); tgt = batch["tg"].to(device)
                vl += level_loss(m.forward_ts_only(tf), tgt).item() * tf.size(0)
                vn += tf.size(0)
        vl /= max(vn, 1)
        log(f"  Loaded cached P1 {p1_path.name}, val={vl:.4f}")
        return clone_ts_state(m), float(vl)
    log(f"  Training P1 (no cache at {p1_path})")
    t0 = time.time()
    ts_state, p1_val, _ = train_p1(loaders, feat_dim, device)
    log(f"  P1 val={p1_val:.4f} [{time.time()-t0:.0f}s]")
    return ts_state, p1_val


def train_p2(loaders, feat_dim, p1_state, device, decay_spec, huber_delta=0.0):
    set_seed(SEED)
    if huber_delta > 0:
        def loss_fn(p, t): return huber_log_loss(p, t, delta=huber_delta)
    else:
        loss_fn = level_loss
    model = CryptoBertMainModelDecay(feat_dim, decay_spec=decay_spec).to(device)
    load_ts_state(model, p1_state)
    for n, p in model.named_parameters():
        p.requires_grad = not n.startswith(("ts_net", "ts_head"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CB_P2_LR)
    best, pat, best_sd = float("inf"), 0, None
    for ep in range(CB_P2_MAX_EPOCHS):
        model.train()
        for batch in loaders["train"]:
            tf = batch["tf"].to(device); nf = batch["nf"].to(device)
            nt = batch["nt"].to(device); bt = batch["bt"].to(device)
            mk = batch["mk"].to(device); tgt = batch["tg"].to(device)
            loss = loss_fn(model(tf, nf, nt, bt, mk), tgt)
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
                vl += loss_fn(model(tf, nf, nt, bt, mk), tgt).item() * tf.size(0)
                vn += tf.size(0)
        vl /= max(vn, 1)
        if vl < best:
            best, pat, best_sd = vl, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if pat >= CB_P2_PATIENCE:
            break
    model.load_state_dict(best_sd)
    return model, float(best)


def predict(model, p1_state, loader, feat_dim, device):
    ts_model = CryptoBertMainModelDecay(feat_dim).to(device)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", nargs="+", type=int, default=list(range(1, len(WINDOWS) + 1)),
                        help="Source-window indices (1..8 in main_model_common.WINDOWS)")
    parser.add_argument("--variants", nargs="+", choices=DECAY_SPECS, default=DECAY_SPECS)
    parser.add_argument("--huber-delta", type=float, default=CB_HUBER_DELTA,
                        help="Huber-on-log(vol) delta. 0 = MSE. Default 0.5 (matches v3).")
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Device: {DEVICE}; horizon: {HORIZON_LABEL}; variants: {args.variants}")
    log(f"Source windows: {args.windows}")
    log(f"Output dir: {out_dir}")
    log(f"Loss: {'Huber-on-log delta=' + str(args.huber_delta) if args.huber_delta > 0 else 'MSE-on-log'}")

    bars = build_price_bars("BTC")
    news = load_news_features("BTC")
    bars, left, right = align_bars_with_news(bars, news["timestamps"])
    cb_emb = load_cryptobert_embeddings("BTC", news["timestamps"])
    log(f"Bars: {len(bars):,}; CryptoBERT: {cb_emb.shape}")

    zero_topic = np.zeros((len(news["timestamps"]), 1), dtype=np.float32)
    bars_dt = bars["datetime"]
    rows = []
    pred_store = {}

    for src_w in args.windows:
        if src_w > len(WINDOWS):
            continue
        val_start, val_end, test_start, test_end = WINDOWS[src_w - 1]
        vs = pd.Timestamp(val_start, tz="UTC"); ve = pd.Timestamp(val_end, tz="UTC")
        ts = pd.Timestamp(test_start, tz="UTC"); te = pd.Timestamp(test_end, tz="UTC")
        train_idx = np.where(bars_dt < vs)[0]
        val_idx = np.where((bars_dt >= vs) & (bars_dt <= ve))[0]
        test_idx = np.where((bars_dt >= ts) & (bars_dt <= te))[0]
        if len(train_idx) < 1000 or len(val_idx) < 100 or len(test_idx) < 100:
            log(f"src_w={src_w}: skipped"); continue

        # Label as production: source_w → window = source_w - 2 (for windows 3..8)
        win_label = src_w - SOURCE_WINDOW_OFFSET if src_w > SOURCE_WINDOW_OFFSET else src_w
        log(""); log("=" * 72)
        log(f"Window label {win_label} (source {src_w}): test {test_start} -> {test_end}")
        log(f"Train={len(train_idx):,} Val={len(val_idx):,} Test={len(test_idx):,}")

        bars_norm, _, _ = normalize_bars(bars, train_idx)
        actual = bars[TARGET_COL].iloc[test_idx].to_numpy(dtype=np.float64)
        loaders = make_split_loaders(
            bars_norm, TARGET_COL, train_idx, val_idx, test_idx, left, right,
            cb_emb, zero_topic, news, CB_BATCH_SIZE,
        )

        p1_state, p1_val = load_or_train_p1(win_label, loaders, cb_emb.shape[1], DEVICE)
        pred_store.setdefault(win_label, {"actual": actual})

        for spec in args.variants:
            log("")
            log(f"  Variant: {spec}")
            t0 = time.time()
            model, p2_val = train_p2(loaders, cb_emb.shape[1], p1_state, DEVICE,
                                       spec, huber_delta=args.huber_delta)
            ts_pred, main_pred = predict(model, p1_state, loaders["test"], cb_emb.shape[1], DEVICE)
            elapsed = time.time() - t0

            ts_m = compute_metrics(ts_pred, actual)
            main_m = compute_metrics(main_pred, actual)
            delta_r2 = main_m["R2"] - ts_m["R2"]
            log(f"    P2 val={p2_val:.4f}  Main R2={main_m['R2']:.4f}  "
                f"TS R2={ts_m['R2']:.4f}  Δ={delta_r2:+.4f}  [{elapsed:.0f}s]")

            pred_store[win_label][spec] = main_pred
            pred_store[win_label]["ts"] = ts_pred  # same across variants
            rows.append({
                "source_window": src_w, "window": win_label,
                "test_period": f"{test_start}->{test_end}",
                "decay_spec": spec, "p1_val": p1_val, "p2_val": p2_val,
                "ts_r2": ts_m["R2"], "main_r2": main_m["R2"], "delta_r2": delta_r2,
                "main_mse": main_m["MSE"], "ts_mse": ts_m["MSE"],
                "main_qlike": main_m["QLIKE"], "ts_qlike": ts_m["QLIKE"],
                "main_da": main_m["DA"], "ts_da": ts_m["DA"],
                "elapsed_s": elapsed, "N": len(actual),
            })
            pd.DataFrame(rows).to_csv(out_dir / "per_window.csv",
                                       index=False, float_format="%.6f")

    detail = pd.DataFrame(rows)
    pooled = (detail.groupby("decay_spec")
                    .agg(R2_mean=("main_r2", "mean"), R2_std=("main_r2", "std"),
                         deltaR2_mean=("delta_r2", "mean"),
                         deltaR2_std=("delta_r2", "std"),
                         MSE_mean=("main_mse", "mean"),
                         QLIKE_mean=("main_qlike", "mean"),
                         DA_mean=("main_da", "mean"),
                         n_windows=("window", "nunique"))
                    .reset_index().sort_values("R2_mean", ascending=False))
    pooled.to_csv(out_dir / "pooled_summary.csv", index=False, float_format="%.6f")
    log("\n=== Pooled summary (sorted by R²) ===")
    log(pooled.to_string(index=False))

    # DM tests vs exponential (the production baseline)
    if "exponential" in detail["decay_spec"].unique() and len(args.windows) >= 2:
        log("\n=== DM tests pooled (variant vs exponential, MSE & QLIKE) ===")
        ws = sorted(pred_store.keys())
        actual_pool = np.concatenate([pred_store[w]["actual"] for w in ws])
        exp_pool = np.concatenate([pred_store[w]["exponential"] for w in ws])
        exp_m = compute_metrics(exp_pool, actual_pool)
        dm_rows = []
        for spec in detail["decay_spec"].unique():
            if spec == "exponential":
                continue
            try:
                pool = np.concatenate([pred_store[w][spec] for w in ws])
            except KeyError:
                continue
            m = compute_metrics(pool, actual_pool)
            t_mse, p_mse = dm_test(m["mse_vec"], exp_m["mse_vec"], 16)
            t_q, p_q = dm_test(m["qlike_vec"], exp_m["qlike_vec"], 16)
            dm_rows.append({
                "decay_spec": spec, "R2": m["R2"], "MSE": m["MSE"], "QLIKE": m["QLIKE"],
                "DM_MSE_t_vs_exp": t_mse, "DM_MSE_p_vs_exp": p_mse,
                "DM_QLIKE_t_vs_exp": t_q, "DM_QLIKE_p_vs_exp": p_q,
            })
        if dm_rows:
            dm_df = pd.DataFrame(dm_rows)
            dm_df.to_csv(out_dir / "dm_vs_exponential.csv", index=False, float_format="%.6f")
            log(dm_df.to_string(index=False))


if __name__ == "__main__":
    main()
