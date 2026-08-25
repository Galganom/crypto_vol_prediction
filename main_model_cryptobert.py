"""
Main pipeline using CryptoBERT-PCA news features with P1/P2 staged training.

Structurally mirrors main_model_common.run_expanding_training but replaces:
  - News features: feat_topic (sentiment+topic_flags, 9-dim FinBERT-derived)
                 -> CryptoBERT embeddings reduced via per-window PCA to k=256
  - Main Model arch: TopicCategoryDecayModel (s_head 9->32->1 + per-topic lambda)
                  -> Per-article CryptoBERT-PCA news head (256->128->32->2)
  - EGARCH+NED arch: same per-article (s, lambda) head over CryptoBERT-PCA
  - Training: batch size 64, P1 TS LR 1e-4, P2 news LR 1e-4,
              P2 is the terminal checkpoint, gradient clipping at 1.0,
              hidden=(128, 32)

HAR, EGARCH baselines and TS MLP are unchanged.
"""
import argparse
import json
import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from main_model_common import (
    AUX_WEIGHT, HORIZONS, NT, P1_SEED, SEED, TC, WINDOWS,
    align_bars_with_news, build_price_bars, cfg, clone_ts_state,
    compute_metrics, dm_test, fit_egarch_all, fit_har,
    get_horizon_cfg, huber_log_loss, level_loss, load_news_features,
    load_ts_state, log, make_split_loaders, normalize_bars, set_seed,
)

# CryptoBERT-PCA configuration.
CB_PCA_K = 256
CB_HIDDEN = (128, 32)
CB_P1_LR = 1e-4            # CryptoBERT run's TS-only P1 LR
CB_P2_LR = 5e-5            # halved for Huber-loss main model; reduces early-stop on W4-like windows
CB_BATCH_SIZE = 64
CB_DEFAULT_HORIZONS = ["4h", "1h"]
CB_SOURCE_WINDOW_OFFSET = 2  # Output window 1 starts at the 2023-01 test split.
CB_DEFAULT_WINDOWS = [1, 2, 3, 4, 5, 6]
CB_GRAD_CLIP = 1.0
CB_AUX_WEIGHT = 0.0        # Aux loss disabled; optimize the primary forecast loss only
CB_P1_MAX_EPOCHS = 100
CB_P2_MAX_EPOCHS = 120
CB_P1_PATIENCE = 15
CB_P2_PATIENCE = 20        # bumped from 15 to give slow-LR runs more room
CB_LOSS_LOG_EVERY = 1
CB_HUBER_DELTA = 0.5       # Huber-on-log(vol) delta; main model. Set 0 to revert to MSE.
CB_TS_SCALE_LR = 0.0       # ts_scale frozen at 1.0 in the main model


def resolve_cryptobert_window_map(window_labels):
    """Map public 2023-based window labels (1..6) to source WINDOWS indices."""
    labels = sorted(set(window_labels))
    bad = [w for w in labels if w < 1 or w > 6]
    if bad:
        raise ValueError(
            f"CryptoBERT windows are labeled 1..6, where 1 is the 2023-01 test split; got {bad}."
        )
    return {w + CB_SOURCE_WINDOW_OFFSET: w for w in labels}


def cryptobert_results_dir(asset, news_asset=None):
    asset = asset.upper()
    news_asset = (news_asset or asset).upper()
    if news_asset == asset:
        return f"results/main/{asset.lower()}_cryptobert"
    return f"results/main/{asset.lower()}_{news_asset.lower()}_news_cryptobert"


class CryptoBertMainModel(nn.Module):
    """Per-article (s, λ) decay news head over CryptoBERT-PCA features.

    News head:
        feat_dim -> hidden dims -> 2 (softplus(s), 0.001 + 0.499*sigmoid(lambda))
    TS branch identical to main_model_common's TopicCategoryDecayModel so we
    can reuse clone_ts_state / load_ts_state.
    """

    def __init__(self, feat_dim, hidden=CB_HIDDEN):
        super().__init__()
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
        # P3 hook: learnable scalar on the TS branch output before the news
        # residual exp() factor. Default 1.0 leaves P1/P2 behavior unchanged
        # (P1/P2 keep it frozen at 1.0; P3 unfreezes it).
        self.ts_scale = nn.Parameter(torch.tensor(1.0))
        self.ts_scale.requires_grad = False

    def _event_signal(self, nf, nt, bt, mk):
        bsz, n_news, feat_dim = nf.shape
        x = self.input_norm(nf.reshape(bsz * n_news, feat_dim))
        out = self.news_head(x).reshape(bsz, n_news, 2)
        s = F.softplus(out[:, :, 0:1])
        lam = 0.001 + 0.499 * torch.sigmoid(out[:, :, 1:2])
        dt = (bt.unsqueeze(1) - nt).float().clamp(min=0) / 60.0
        return (s * torch.exp(-lam * dt.unsqueeze(2)) * mk.unsqueeze(2).float()).sum(1)

    def forward_ts_only(self, tf):
        return self.ts_head(self.ts_net(tf))

    def forward(self, tf, nf, nt, bt, mk, tm=None):
        base = self.ts_scale * self.forward_ts_only(tf)
        signal = self._event_signal(nf, nt, bt, mk)
        return base * torch.exp(self.residual_head(signal).clamp(-2, 2))

    def forward_aux(self, nf, nt, bt, mk, tm=None):
        return self.aux_head(self._event_signal(nf, nt, bt, mk))


class CryptoBertEgarchNedModel(nn.Module):
    """EGARCH+NED with the same CryptoBERT news head as the main model."""

    def __init__(self, feat_dim, hidden=CB_HIDDEN):
        super().__init__()
        self.input_norm = nn.LayerNorm(feat_dim)
        self.news_head = make_news_head(feat_dim, hidden)
        self.residual_head = nn.Linear(1, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.aux_head = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Softplus(),
        )

    def _event_signal(self, nf, nt, bt, mk):
        bsz, n_news, feat_dim = nf.shape
        x = self.input_norm(nf.reshape(bsz * n_news, feat_dim))
        out = self.news_head(x).reshape(bsz, n_news, 2)
        s = F.softplus(out[:, :, 0:1])
        lam = 0.001 + 0.499 * torch.sigmoid(out[:, :, 1:2])
        dt = (bt.unsqueeze(1) - nt).float().clamp(min=0) / 60.0
        return (s * torch.exp(-lam * dt.unsqueeze(2)) * mk.unsqueeze(2).float()).sum(1)

    def forward(self, base, nf, nt, bt, mk, tm=None):
        signal = self._event_signal(nf, nt, bt, mk)
        return base * torch.exp(self.residual_head(signal).clamp(-2, 2))

    def forward_aux(self, nf, nt, bt, mk, tm=None):
        return self.aux_head(self._event_signal(nf, nt, bt, mk))


def make_news_head(feat_dim, hidden):
    if isinstance(hidden, int):
        hidden_dims = [hidden]
    else:
        hidden_dims = list(hidden)
    layers = []
    in_dim = feat_dim
    for out_dim in hidden_dims:
        layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, 2))
    return nn.Sequential(*layers)


def load_cryptobert_embeddings(asset, news_timestamps):
    """Load and sort CryptoBERT embeddings to align with news_timestamps order."""
    c = cfg(asset)
    path = c.get("emb_cryptobert")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"{asset} CryptoBERT embeddings not found at {path}; "
            f"run step2_eth_cryptobert.py / step2_btc_cryptobert.py first."
        )
    data = np.load(path)
    si = np.argsort(data["timestamps"])
    emb = data["embeddings"][si].astype(np.float32)
    if not np.array_equal(data["timestamps"][si], news_timestamps):
        raise ValueError(
            f"{asset} CryptoBERT timestamps do not align with news timestamps."
        )
    return emb


def fit_cryptobert_pca(emb_full, train_mask, k=CB_PCA_K, seed=SEED):
    """Fit PCA on training-period CryptoBERT embeddings; transform full corpus."""
    pca = PCA(n_components=k, random_state=seed)
    pca.fit(emb_full[train_mask])
    return pca.transform(emb_full).astype(np.float32), pca


def _save_loss_history(history, csv_path=None, plot_path=None):
    if not history or not csv_path:
        return
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv(csv_path, index=False, float_format="%.10f")
    if not plot_path:
        return
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for phase, phase_df in df.groupby("phase", sort=False):
            ax.plot(
                phase_df["global_epoch"], phase_df["train_loss"],
                marker="o", markersize=2.5, linewidth=1.2,
                label=f"{phase} train",
            )
            ax.plot(
                phase_df["global_epoch"], phase_df["val_loss"],
                marker="o", markersize=2.5, linewidth=1.2,
                linestyle="--", label=f"{phase} val",
            )
        ax.set_xlabel("training epoch")
        ax.set_ylabel("log-volatility loss")
        ax.set_title("CryptoBERT Main Model loss")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        plot_dir = os.path.dirname(plot_path)
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
    except Exception as exc:
        log(f"    Loss plot skipped: {exc}")


def _record_epoch(
    history, phase, epoch, global_epoch, train_loss, train_primary_loss,
    val_loss, best, patience, lr_news, lr_ts, improved, elapsed_s,
):
    history.append({
        "phase": phase,
        "epoch": epoch,
        "global_epoch": global_epoch,
        "train_loss": float(train_loss),
        "train_primary_loss": float(train_primary_loss),
        "val_loss": float(val_loss),
        "best_val_loss": float(best),
        "patience": int(patience),
        "lr_news": float(lr_news) if lr_news is not None else np.nan,
        "lr_ts": float(lr_ts) if lr_ts is not None else np.nan,
        "improved": bool(improved),
        "elapsed_s": float(elapsed_s),
    })


def _log_loss_epoch(
    phase, epoch, train_loss, train_primary_loss, val_loss, best, patience,
    lr_news=None, lr_ts=None, improved=False, loss_log_every=CB_LOSS_LOG_EVERY,
):
    if loss_log_every <= 0:
        return
    if epoch % loss_log_every != 0 and not improved:
        return
    marker = " *" if improved else ""
    lr_bits = []
    if lr_news is not None:
        lr_bits.append(f"news_lr={lr_news:.2e}")
    if lr_ts is not None:
        lr_bits.append(f"ts_lr={lr_ts:.2e}")
    lr_text = f"  {' '.join(lr_bits)}" if lr_bits else ""
    primary_text = ""
    if abs(train_loss - train_primary_loss) > 1e-12:
        primary_text = f" train_primary={train_primary_loss:.4f}"
    log(
        f"    {phase} ep {epoch:3d}  train={train_loss:.4f}"
        f"{primary_text}  val={val_loss:.4f}  best={best:.4f}"
        f"  pat={patience}{lr_text}{marker}"
    )


def train_cryptobert_main_model(loaders, feat_dim, device,
                                 p1_lr=CB_P1_LR, p2_lr=CB_P2_LR,
                                 ts_scale_lr=1e-3,
                                 hidden=CB_HIDDEN,
                                 p1_max_epochs=CB_P1_MAX_EPOCHS,
                                 p2_max_epochs=CB_P2_MAX_EPOCHS,
                                 p1_patience=CB_P1_PATIENCE,
                                 p2_patience=CB_P2_PATIENCE,
                                 aux_weight=CB_AUX_WEIGHT,
                                 loss_log_every=CB_LOSS_LOG_EVERY,
                                 loss_csv_path=None,
                                 loss_plot_path=None,
                                 shared_p1_state=None,
                                 model_seed=P1_SEED,
                                 train_seed=SEED,
                                 huber_delta=0.0):
    """P1 trains the TS branch; P2 trains the news residual + ts_scale (TS frozen).

    If huber_delta > 0, both phases use Huber-on-log loss with that delta
    instead of plain MSE-on-log; this down-weights tail-vol bars.
    """
    if huber_delta > 0:
        def loss_fn(pred, target):
            return huber_log_loss(pred, target, delta=huber_delta)
    else:
        loss_fn = level_loss
    set_seed(model_seed)
    model = CryptoBertMainModel(feat_dim, hidden=hidden).to(device)
    history = []
    global_epoch = 0
    train_start = time.time()

    if shared_p1_state is None:
        # P1: train TS-only; freeze news head + residual + aux + ts_scale
        for p in model.news_head.parameters(): p.requires_grad = False
        for p in model.residual_head.parameters(): p.requires_grad = False
        for p in model.aux_head.parameters(): p.requires_grad = False
        model.ts_scale.requires_grad = False
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=p1_lr)
        best, pat, p1_sd = float("inf"), 0, None
        for ep in range(p1_max_epochs):
            model.train()
            train_sum, train_n = 0.0, 0
            for batch in loaders["train"]:
                tf = batch["tf"].to(device); tgt = batch["tg"].to(device)
                loss = loss_fn(model.forward_ts_only(tf), tgt)
                opt.zero_grad(); loss.backward(); opt.step()
                train_sum += loss.item() * tf.size(0)
                train_n += tf.size(0)
            train_loss = train_sum / max(train_n, 1)
            model.eval()
            vl, vn = 0.0, 0
            with torch.no_grad():
                for batch in loaders["val"]:
                    tf = batch["tf"].to(device); tgt = batch["tg"].to(device)
                    vl += loss_fn(model.forward_ts_only(tf), tgt).item() * tf.size(0)
                    vn += tf.size(0)
            vl /= max(vn, 1)
            improved = vl < best
            if improved:
                best, pat = vl, 0
                p1_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                pat += 1
            global_epoch += 1
            _record_epoch(
                history, "P1", ep + 1, global_epoch, train_loss, train_loss,
                vl, best, pat, None, p1_lr, improved, time.time() - train_start,
            )
            _log_loss_epoch(
                "P1", ep + 1, train_loss, train_loss, vl, best, pat,
                lr_ts=p1_lr, improved=improved, loss_log_every=loss_log_every,
            )
            if loss_csv_path:
                _save_loss_history(history, loss_csv_path)
            if pat >= p1_patience:
                break
        if p1_sd is None:
            raise RuntimeError("P1 did not run; set --p1-max-epochs to at least 1.")
        model.load_state_dict(p1_sd)
        log(f"    P1 best val: {best:.4f}")
    else:
        load_ts_state(model, shared_p1_state)
        log("    P1 shared TS state loaded; skipping P1")
    p1_full_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    pure_p1_state = clone_ts_state(model)

    # P2: freeze TS, train news head + residual + aux + (optionally) ts_scale.
    set_seed(train_seed)
    for p in model.parameters(): p.requires_grad = True
    for p in model.ts_net.parameters(): p.requires_grad = False
    for p in model.ts_head.parameters(): p.requires_grad = False
    if ts_scale_lr > 0:
        model.ts_scale.requires_grad = True
        p2_other = [p for n, p in model.named_parameters()
                    if p.requires_grad and n != "ts_scale"]
        opt = torch.optim.AdamW(
            [{"params": p2_other, "lr": p2_lr},
             {"params": [model.ts_scale], "lr": ts_scale_lr}]
        )
    else:
        model.ts_scale.requires_grad = False
        p2_other = [p for n, p in model.named_parameters() if p.requires_grad]
        opt = torch.optim.AdamW(p2_other, lr=p2_lr)
    best, pat, p2_sd = float("inf"), 0, None
    for ep in range(p2_max_epochs):
        model.train()
        train_total_sum, train_primary_sum, train_n = 0.0, 0.0, 0
        for batch in loaders["train"]:
            tf = batch["tf"].to(device); nf = batch["nf"].to(device)
            nt = batch["nt"].to(device); bt = batch["bt"].to(device)
            mk = batch["mk"].to(device); tgt = batch["tg"].to(device)
            primary_loss = loss_fn(model(tf, nf, nt, bt, mk), tgt)
            if aux_weight > 0:
                aux_loss = loss_fn(model.forward_aux(nf, nt, bt, mk), tgt)
                loss = primary_loss + aux_weight * aux_loss
            else:
                loss = primary_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CB_GRAD_CLIP)
            opt.step()
            with torch.no_grad():
                model.ts_scale.clamp_(min=1e-3)
            train_total_sum += loss.item() * tf.size(0)
            train_primary_sum += primary_loss.item() * tf.size(0)
            train_n += tf.size(0)
        train_loss = train_total_sum / max(train_n, 1)
        train_primary_loss = train_primary_sum / max(train_n, 1)
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
        improved = vl < best
        if improved:
            best, pat = vl, 0
            p2_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        global_epoch += 1
        _record_epoch(
            history, "P2", ep + 1, global_epoch, train_loss, train_primary_loss,
            vl, best, pat, p2_lr, 0.0, improved, time.time() - train_start,
        )
        ts_scale_val = float(model.ts_scale.detach().cpu().item())
        history[-1]["ts_scale"] = ts_scale_val
        _log_loss_epoch(
            "P2", ep + 1, train_loss, train_primary_loss, vl, best, pat,
            lr_news=p2_lr, lr_ts=ts_scale_lr, improved=improved,
            loss_log_every=loss_log_every,
        )
        if loss_log_every > 0 and (ep + 1) % loss_log_every == 0:
            log(f"      ts_scale={ts_scale_val:.4f}")
        if loss_csv_path:
            _save_loss_history(history, loss_csv_path)
        if pat >= p2_patience:
            break
    if p2_sd is None:
        raise RuntimeError("P2 did not run; set --p2-max-epochs to at least 1.")
    p2_best_val = float(best)
    log(f"    P2 best val: {p2_best_val:.4f}")
    model.load_state_dict(p2_sd)
    _save_loss_history(history, loss_csv_path, loss_plot_path)
    return model, pure_p1_state, p1_full_state


def predict_main_and_ts(model, p1_state, loader, feat_dim, device):
    """TS-only predictions use the canonical P1 state; full predictions use the
    P2 news-residual model."""
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


def train_cryptobert_egarch_ned(loaders, egarch_train, egarch_val, egarch_test, device,
                                 hidden=CB_HIDDEN, lr=2e-4, huber_delta=0.0):
    """Train EGARCH+NED with CryptoBERT-PCA news head.

    If huber_delta > 0, uses Huber-on-log(vol) loss to match the main model.
    """
    set_seed(SEED)
    feat_dim = loaders["train"].dataset.feat_dim
    model = CryptoBertEgarchNedModel(feat_dim, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_base = torch.tensor(egarch_train.reshape(-1, 1), dtype=torch.float32)
    val_base = torch.tensor(egarch_val.reshape(-1, 1), dtype=torch.float32)
    if huber_delta > 0:
        def loss_fn(p, t): return huber_log_loss(p, t, delta=huber_delta)
    else:
        loss_fn = level_loss
    best, pat, best_sd = float("inf"), 0, None
    for ep in range(120):
        model.train()
        cursor = 0
        for batch in loaders["train"]:
            n = batch["tg"].shape[0]
            base = train_base[cursor:cursor + n].to(device); cursor += n
            nf = batch["nf"].to(device); nt = batch["nt"].to(device)
            bt = batch["bt"].to(device); mk = batch["mk"].to(device)
            tgt = batch["tg"].to(device)
            loss = loss_fn(model(base, nf, nt, bt, mk), tgt)
            loss = loss + AUX_WEIGHT * loss_fn(model.forward_aux(nf, nt, bt, mk), tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CB_GRAD_CLIP)
            opt.step()
        model.eval()
        vl, vn, cursor = 0.0, 0, 0
        with torch.no_grad():
            for batch in loaders["val"]:
                n = batch["tg"].shape[0]
                base = val_base[cursor:cursor + n].to(device); cursor += n
                nf = batch["nf"].to(device); nt = batch["nt"].to(device)
                bt = batch["bt"].to(device); mk = batch["mk"].to(device)
                tgt = batch["tg"].to(device)
                vl += loss_fn(model(base, nf, nt, bt, mk), tgt).item() * n
                vn += n
        vl /= max(vn, 1)
        if vl < best:
            best, pat = vl, 0
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if pat >= 20:
            break
    model.load_state_dict(best_sd)
    model.eval()
    test_base = torch.tensor(egarch_test.reshape(-1, 1), dtype=torch.float32)
    preds, cursor = [], 0
    with torch.no_grad():
        for batch in loaders["test"]:
            n = batch["tg"].shape[0]
            base = test_base[cursor:cursor + n].to(device); cursor += n
            nf = batch["nf"].to(device); nt = batch["nt"].to(device)
            bt = batch["bt"].to(device); mk = batch["mk"].to(device)
            preds.append(model(base, nf, nt, bt, mk).cpu().numpy())
    return model, np.concatenate(preds).squeeze(1)


def run_cryptobert_expanding_training(
    asset, results_dir=None, batch_size=CB_BATCH_SIZE, run_dm=True,
    rebuild_topics=False, news_asset=None, horizons=None, windows=None,
    p1_lr=CB_P1_LR, p2_lr=CB_P2_LR, ts_scale_lr=1e-3, hidden=CB_HIDDEN,
    p1_max_epochs=CB_P1_MAX_EPOCHS, p2_max_epochs=CB_P2_MAX_EPOCHS,
    p1_patience=CB_P1_PATIENCE, p2_patience=CB_P2_PATIENCE,
    loss_log_every=CB_LOSS_LOG_EVERY, loss_plot=True,
    aux_weight=CB_AUX_WEIGHT, skip_egarch_ned=False,
    p1_checkpoints_dir=None, huber_delta=0.0,
):
    asset = asset.upper()
    news_asset = (news_asset or asset).upper()
    horizon_names = list(horizons or CB_DEFAULT_HORIZONS)
    window_map = resolve_cryptobert_window_map(windows or CB_DEFAULT_WINDOWS)
    window_filter = set(window_map)
    cfg(asset); cfg(news_asset)
    results_dir = results_dir or cryptobert_results_dir(asset, news_asset)
    for sub in ["checkpoints", "predictions", "metadata", "loss_curves"]:
        os.makedirs(os.path.join(results_dir, sub), exist_ok=True)

    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"Price asset: {asset}; News asset: {news_asset}")
    log(f"Results dir: {results_dir}")
    log(f"Windows: {sorted(window_map.values())} (2023-based labels; source windows {sorted(window_filter)})")
    log(f"Horizons: {', '.join(horizon_names)}")
    log(f"CryptoBERT full 768-dim (no PCA), hidden={hidden}, batch_size={batch_size}")
    log(f"LRs: P1={p1_lr:g}, P2 news={p2_lr:g}, P2 ts_scale={ts_scale_lr:g}")
    if huber_delta > 0:
        log(f"Loss: Huber-on-log(vol), delta={huber_delta:g}")
    else:
        log("Loss: MSE-on-log(vol)")
    if p1_checkpoints_dir:
        log(f"P1 checkpoints loaded from: {p1_checkpoints_dir} (P1 phase skipped)")
    if aux_weight > 0:
        log(f"Loss: primary + aux_weight*aux, aux_weight={aux_weight:g}")
    else:
        log("Loss: primary only (aux disabled)")
    log(
        "Epoch caps: "
        f"P1={p1_max_epochs}/pat{p1_patience}, "
        f"P2={p2_max_epochs}/pat{p2_patience}"
    )
    if skip_egarch_ned:
        log("EGARCH+NED training: skipped")

    bars = build_price_bars(asset)
    news = load_news_features(news_asset, rebuild_topics=rebuild_topics)
    bars, left, right = align_bars_with_news(bars, news["timestamps"])
    log(f"Bars after news overlap: {len(bars):,}")

    cb_emb = load_cryptobert_embeddings(news_asset, news["timestamps"])
    log(f"CryptoBERT embeddings: {cb_emb.shape}")

    rows = []
    bars_dt = bars["datetime"]
    n_articles = len(news["timestamps"])
    zero_topic = np.zeros((n_articles, 1), dtype=np.float32)

    for source_window_num, (val_start, val_end, test_start, test_end) in enumerate(WINDOWS, start=1):
        if source_window_num not in window_filter:
            continue
        window_num = window_map[source_window_num]
        vs = pd.Timestamp(val_start, tz="UTC"); ve = pd.Timestamp(val_end, tz="UTC")
        ts = pd.Timestamp(test_start, tz="UTC"); te = pd.Timestamp(test_end, tz="UTC")
        train_idx = np.where(bars_dt < vs)[0]
        val_idx = np.where((bars_dt >= vs) & (bars_dt <= ve))[0]
        test_idx = np.where((bars_dt >= ts) & (bars_dt <= te))[0]
        if len(train_idx) < 1000 or len(val_idx) < 100 or len(test_idx) < 100:
            log(f"Window {window_num}: skipped"); continue

        log(""); log("=" * 72)
        log(f"Window {window_num}/6: test {test_start} -> {test_end} (source window {source_window_num})")
        log(f"Train={len(train_idx):,} Val={len(val_idx):,} Test={len(test_idx):,}")
        log("=" * 72)

        bars_norm, norm_mean, norm_std = normalize_bars(bars, train_idx)
        np.savez(
            os.path.join(results_dir, "metadata", f"window_{window_num}_meta.npz"),
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            norm_mean=norm_mean, norm_std=norm_std,
        )

        # Use FULL CryptoBERT 768-dim embeddings directly (no PCA reduction).
        # The Main Model has a LayerNorm at input to handle the heterogeneous
        # scale of the raw embeddings.
        cb_pca_full = cb_emb  # alias retained for downstream code; not a PCA
        train_news_mask = (news["timestamps"] >= bars["timestamp"].iloc[train_idx[0]]) & \
                          (news["timestamps"] <= bars["timestamp"].iloc[train_idx[-1]])
        log(f"  CryptoBERT raw 768-dim: {train_news_mask.sum():,} train articles "
            f"(no PCA reduction)")

        start = time.time()
        egarch_base_15m = fit_egarch_all(bars, train_idx)
        log(f"  EGARCH fit/filter: {time.time() - start:.0f}s")

        for horizon in horizon_names:
            h_cfg = get_horizon_cfg(horizon)
            target_col = h_cfg["target"]
            egarch_all = np.clip(egarch_base_15m * h_cfg["bars"], 1e-15, None)
            log(f"\n  Horizon {horizon} ({target_col})")

            har_pred = fit_har(bars, train_idx, test_idx, target_col)
            main_loaders = make_split_loaders(
                bars_norm, target_col, train_idx, val_idx, test_idx, left, right,
                cb_pca_full, zero_topic, news, batch_size,
            )
            t0 = time.time()
            loss_csv_path = os.path.join(
                results_dir, "loss_curves",
                f"main_model_w{window_num}_{horizon}_loss.csv",
            )
            loss_plot_path = os.path.join(
                results_dir, "loss_curves",
                f"main_model_w{window_num}_{horizon}_loss.png",
            ) if loss_plot else None
            shared_p1_state = None
            if p1_checkpoints_dir:
                p1_path = os.path.join(
                    p1_checkpoints_dir, f"p1_model_w{window_num}_{horizon}.pt"
                )
                if os.path.exists(p1_path):
                    raw = torch.load(p1_path, map_location=device)
                    shared_p1_state = {
                        k: v for k, v in raw.items()
                        if k.startswith("ts_net") or k.startswith("ts_head")
                    }
                    log(f"  Loaded P1 checkpoint (TS-branch only): {p1_path}")
                else:
                    log(f"  P1 checkpoint not found at {p1_path}; "
                        f"will train P1 from scratch for this horizon.")
            main_model, p1_state, p1_full_state = train_cryptobert_main_model(
                main_loaders, cb_pca_full.shape[1], device,
                p1_lr=p1_lr, p2_lr=p2_lr, ts_scale_lr=ts_scale_lr, hidden=hidden,
                p1_max_epochs=p1_max_epochs, p2_max_epochs=p2_max_epochs,
                p1_patience=p1_patience, p2_patience=p2_patience,
                aux_weight=aux_weight,
                loss_log_every=loss_log_every,
                loss_csv_path=loss_csv_path, loss_plot_path=loss_plot_path,
                shared_p1_state=shared_p1_state,
                huber_delta=huber_delta,
            )
            ts_scale_learned = float(main_model.ts_scale.detach().cpu().item())
            log(f"  ts_scale learned: {ts_scale_learned:.4f}")
            log(f"  Main Model trained [{time.time() - t0:.0f}s]")
            log(f"  Main Model loss curve: {loss_csv_path}")
            if loss_plot_path:
                log(f"  Main Model loss plot: {loss_plot_path}")

            ts_pred, main_pred = predict_main_and_ts(
                main_model, p1_state, main_loaders["test"],
                cb_pca_full.shape[1], device,
            )

            actual = bars[target_col].iloc[test_idx].to_numpy(dtype=np.float64)
            if skip_egarch_ned:
                egarch_ned_model = None
                egarch_ned_pred = np.full_like(actual, np.nan, dtype=np.float64)
                log("  EGARCH+NED skipped")
            else:
                ned_loaders = make_split_loaders(
                    bars_norm, target_col, train_idx, val_idx, test_idx, left, right,
                    cb_pca_full, zero_topic, news, batch_size, train_shuffle=False,
                )
                t0 = time.time()
                egarch_ned_model, egarch_ned_pred = train_cryptobert_egarch_ned(
                    ned_loaders, egarch_all[train_idx], egarch_all[val_idx], egarch_all[test_idx], device,
                    huber_delta=huber_delta,
                )
                log(f"  EGARCH+NED trained [{time.time() - t0:.0f}s]")
            preds = OrderedDict([
                ("HAR", har_pred),
                ("EGARCH", egarch_all[test_idx]),
            ])
            if not skip_egarch_ned:
                preds["EGARCH+NED"] = egarch_ned_pred
            preds["TS MLP"] = ts_pred
            preds["Main Model"] = main_pred
            np.savez(
                os.path.join(results_dir, "predictions", f"window_{window_num}_{horizon}_predictions.npz"),
                actual=actual, har=har_pred, egarch=egarch_all[test_idx],
                egarch_ned=egarch_ned_pred, ts_mlp=ts_pred, main_model=main_pred,
                test_idx=test_idx, datetime=bars["datetime"].iloc[test_idx].astype(str).values,
                horizon=horizon, target_col=target_col, asset=asset, news_asset=news_asset,
                ts_scale=ts_scale_learned,
            )
            torch.save(main_model.state_dict(),
                       os.path.join(results_dir, "checkpoints", f"main_model_w{window_num}_{horizon}.pt"))
            torch.save(p1_full_state,
                       os.path.join(results_dir, "checkpoints", f"p1_model_w{window_num}_{horizon}.pt"))
            torch.save(p1_state,
                       os.path.join(results_dir, "checkpoints", f"ts_mlp_w{window_num}_{horizon}.pt"))
            if egarch_ned_model is not None:
                torch.save(egarch_ned_model.state_dict(),
                           os.path.join(results_dir, "checkpoints", f"egarch_ned_w{window_num}_{horizon}.pt"))

            for name, pred in preds.items():
                m = compute_metrics(pred, actual)
                rows.append({
                    "asset": asset, "window": window_num, "test_period": f"{test_start}->{test_end}",
                    "news_asset": news_asset, "horizon": horizon, "model": name,
                    "MSE": m["MSE"], "QLIKE": m["QLIKE"], "R2": m["R2"], "DA": m["DA"], "N": len(actual),
                })
                log(f"    {name:<12s} R2={m['R2']:.4f} MSE={m['MSE']:.5f} QLIKE={m['QLIKE']:.5f} DA={m['DA']:.3f}")

    metrics_path = os.path.join(results_dir, "expanding_window_metrics.csv")
    pd.DataFrame(rows).to_csv(metrics_path, index=False, float_format="%.10f")
    log(f"Saved: {metrics_path}")
    if rows:
        mean_df = (
            pd.DataFrame(rows).groupby(["asset", "news_asset", "horizon", "model"], as_index=False)
            .agg(MSE=("MSE", "mean"), QLIKE=("QLIKE", "mean"), R2=("R2", "mean"),
                 DA=("DA", "mean"), N=("N", "sum"), n_windows=("window", "nunique"))
        )
        mean_path = os.path.join(results_dir, "expanding_window_mean_metrics.csv")
        mean_df.to_csv(mean_path, index=False, float_format="%.10f")
        log(f"Saved: {mean_path}")
    with open(os.path.join(results_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "asset": asset, "news_asset": news_asset, "horizons": horizon_names,
            "windows": sorted(window_map.values()),
            "source_windows": sorted(window_filter),
            "window_label_offset": CB_SOURCE_WINDOW_OFFSET,
            "lr_p1": p1_lr, "lr_p2": p2_lr, "lr_ts_scale": ts_scale_lr,
            "p1_checkpoints_dir": p1_checkpoints_dir,
            "ts_scale_learnable": True,
            "cryptobert_pca_k": None, "cryptobert_input_dim": 768, "hidden": hidden,
            "batch_size": batch_size, "grad_clip": CB_GRAD_CLIP,
            "p1_max_epochs": p1_max_epochs,
            "p2_max_epochs": p2_max_epochs,
            "p1_patience": p1_patience,
            "p2_patience": p2_patience,
            "aux_weight": aux_weight,
            "loss_log_every": loss_log_every,
            "loss_plot": loss_plot,
            "skip_egarch_ned": skip_egarch_ned,
            "models": (
                ["HAR", "EGARCH", "TS MLP", "Main Model"]
                if skip_egarch_ned
                else ["HAR", "EGARCH", "EGARCH+NED", "TS MLP", "Main Model"]
            ),
        }, f, indent=2)

    if run_dm and skip_egarch_ned:
        log("DM evaluation skipped because EGARCH+NED was skipped.")
    elif run_dm:
        from main_model_common import run_dm_evaluation
        run_dm_evaluation(asset, results_dir, news_asset=news_asset, horizons=horizon_names)


def parse_main_args(default_asset, default_news_asset=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=CB_BATCH_SIZE)
    parser.add_argument("--skip-dm", action="store_true")
    parser.add_argument("--dm-only", action="store_true")
    parser.add_argument("--rebuild-topics", action="store_true")
    parser.add_argument("--news-asset", default=default_news_asset)
    parser.add_argument("--horizons", nargs="+", choices=list(HORIZONS.keys()), default=None)
    parser.add_argument("--windows", nargs="+", type=int, default=None,
                        help="2023-based window labels 1..6; 1 = 2023-01-01..2023-06-30 test.")
    parser.add_argument("--p1-lr", type=float, default=CB_P1_LR)
    parser.add_argument("--p2-lr", type=float, default=CB_P2_LR)
    parser.add_argument("--ts-scale-lr", type=float, default=CB_TS_SCALE_LR,
                        help="LR for the learnable ts_scale scalar (P2 phase). "
                             "0 = freeze at 1.0 (main model default).")
    parser.add_argument("--p1-checkpoints-dir", default=None,
                        help="If set, load p1_model_w*_*.pt from this dir and skip P1.")
    parser.add_argument("--huber-delta", type=float, default=CB_HUBER_DELTA,
                        help="If >0, use Huber-on-log(vol) loss with this delta "
                             "instead of MSE. delta in log-vol units. "
                             "Main model default: 0.5.")
    parser.add_argument("--hidden", type=int, nargs="+", default=list(CB_HIDDEN),
                        help="News head hidden dims, e.g. --hidden 256 64.")
    parser.add_argument("--p1-max-epochs", type=int, default=CB_P1_MAX_EPOCHS)
    parser.add_argument("--p2-max-epochs", type=int, default=CB_P2_MAX_EPOCHS)
    parser.add_argument("--p1-patience", type=int, default=CB_P1_PATIENCE)
    parser.add_argument("--p2-patience", type=int, default=CB_P2_PATIENCE)
    parser.add_argument("--aux-weight", type=float, default=CB_AUX_WEIGHT)
    parser.add_argument("--loss-log-every", type=int, default=CB_LOSS_LOG_EVERY)
    parser.add_argument("--no-loss-plot", action="store_true")
    parser.add_argument("--skip-egarch-ned", action="store_true")
    args = parser.parse_args()
    if args.dm_only:
        from main_model_common import run_dm_evaluation
        asset = default_asset.upper()
        news_asset = (args.news_asset or default_news_asset or asset).upper()
        results_dir = args.results_dir or cryptobert_results_dir(asset, news_asset)
        run_dm_evaluation(asset, results_dir, news_asset=news_asset, horizons=args.horizons)
    else:
        run_cryptobert_expanding_training(
            default_asset, args.results_dir, args.batch_size, not args.skip_dm,
            args.rebuild_topics, news_asset=args.news_asset,
            horizons=args.horizons, windows=args.windows,
            p1_lr=args.p1_lr, p2_lr=args.p2_lr,
            ts_scale_lr=args.ts_scale_lr, hidden=tuple(args.hidden),
            p1_max_epochs=args.p1_max_epochs,
            p2_max_epochs=args.p2_max_epochs,
            p1_patience=args.p1_patience, p2_patience=args.p2_patience,
            aux_weight=args.aux_weight,
            loss_log_every=args.loss_log_every,
            loss_plot=not args.no_loss_plot, skip_egarch_ned=args.skip_egarch_ned,
            p1_checkpoints_dir=args.p1_checkpoints_dir,
            huber_delta=args.huber_delta,
        )
