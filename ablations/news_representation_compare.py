"""News representation comparison — BTC 4h, 6 expanding windows.

Evaluates 8 ways of feeding news into a residual head on top of a shared
TS-only P1 baseline:

A. WITHOUT temporal decay  (per-bar features, deterministic aggregations)
   1. ts_only                 (no news at all → predicts via TS MLP only)
   2. daily_sentiment         (mean of sentiment_probs over articles
                               in current calendar day so far)
   3. news_count              (count of articles in 12hr lookback)
   4. sentiment_recency       (sentiment_probs weighted by 1/(1+age_minutes))
   5. cryptobert_pca256_mean  (mean of per-article PCA-256 features
                               over articles in 12hr lookback)

B. WITH learned (s, λ) decay  (per-article aggregation; s and λ are learned)
   6. sentiment_decay         (per-article s, λ from sentiment_probs)
   7. cryptobert768_decay     (per-article s, λ from raw 768-dim CryptoBERT)
   8. cryptobert_pca256_decay (per-article s, λ from PCA-256)

All variants share the same shared P1 TS-only state per window. P2 trains the
news head only with TS frozen. P3 is skipped (we want a clean apples-to-apples
news contribution comparison).

Uses the new tag-based nonreactive filter (RECAP_TAGS + SAFE_RECAP_TITLE_PATTERNS).
"""
import argparse
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
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from main_model_common import (  # noqa: E402
    AUX_WEIGHT, LR_P1, NT, P1_SEED, SEED, TC, WINDOWS, LOG_EPS,
    align_bars_with_news, build_price_bars, cfg, clone_ts_state,
    compute_metrics, dm_test, get_horizon_cfg, huber_log_loss, level_loss,
    load_news_features, load_ts_state, log, normalize_bars, set_seed,
)

# ─── Config ──────────────────────────────────────────────────────────────────
HORIZON = "4h"
TARGET_COL = "RV_4h"
LOOKBACK_SECONDS = 12 * 3600
BATCH_SIZE = 64
P1_LR = 1e-4
P2_LR = 5e-5             # match v3 main model
P2_PATIENCE = 20         # match v3 main model
P2_MAX_EPOCHS = 80
GRAD_CLIP = 1.0
DEFAULT_HUBER_DELTA = 0.5  # match v3 main model loss

CRYPTOBERT_PATH = REPO_ROOT / "data/processed/btc_news_embeddings_cryptobert.npz"
FINBERT_CLS_PATH = REPO_ROOT / "data/processed/btc_news_embeddings_finbert_cls.npz"
CRYPTOBERT_SENT_PATH = REPO_ROOT / "data/results/btc_sentiment_scores.parquet"
OUT_DIR = REPO_ROOT / "results/news_repr_compare_btc_4h"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═════════════════════════════════════════════════════════════════════════════
# TS-only model
# ═════════════════════════════════════════════════════════════════════════════
class TSOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ts_net = nn.Sequential(
            nn.Linear(NT, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.ts_head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Softplus(),
        )

    def forward(self, tf):
        return self.ts_head(self.ts_net(tf))


# ═════════════════════════════════════════════════════════════════════════════
# Per-bar feature residual model (variants 2-5: no temporal decay)
# ═════════════════════════════════════════════════════════════════════════════
class BarFeatureResidual(nn.Module):
    """output = ts_pred * exp(residual_head(news_feat).clamp(-2, 2))
    where news_feat is a deterministic per-bar feature (no learned decay)."""

    def __init__(self, feat_dim, hidden=16):
        super().__init__()
        self.ts_net = nn.Sequential(
            nn.Linear(NT, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.ts_head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Softplus(),
        )
        if feat_dim == 1:
            self.news_head = nn.Linear(1, 1)
            nn.init.zeros_(self.news_head.weight); nn.init.zeros_(self.news_head.bias)
        else:
            self.news_head = nn.Sequential(
                nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1),
            )
            nn.init.zeros_(self.news_head[-1].weight); nn.init.zeros_(self.news_head[-1].bias)

    def forward_ts_only(self, tf):
        return self.ts_head(self.ts_net(tf))

    def forward(self, tf, bar_feat):
        base = self.forward_ts_only(tf)
        residual = self.news_head(bar_feat).clamp(-2, 2)
        return base * torch.exp(residual)


# ═════════════════════════════════════════════════════════════════════════════
# Per-article (s, λ) decay residual model (variants 6-8)
# ═════════════════════════════════════════════════════════════════════════════
class PerArticleDecayResidual(nn.Module):
    """News head: feat_dim → hidden → 2 (per-article s, λ).
    Event signal: Σ s_i exp(-λ_i dt_i) over non-reactive articles in lookback."""

    def __init__(self, feat_dim, hidden=16, normalize_input=False):
        super().__init__()
        self.ts_net = nn.Sequential(
            nn.Linear(NT, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.ts_head = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Softplus(),
        )
        self.normalize_input = normalize_input
        if normalize_input:
            self.input_norm = nn.LayerNorm(feat_dim)
        self.news_head = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2),
        )
        self.residual_head = nn.Linear(1, 1)
        nn.init.zeros_(self.residual_head.weight); nn.init.zeros_(self.residual_head.bias)

    def _signal(self, nf, nt, bt, mk):
        bsz, n_news, feat_dim = nf.shape
        x = nf.reshape(bsz * n_news, feat_dim)
        if self.normalize_input:
            x = self.input_norm(x)
        out = self.news_head(x).reshape(bsz, n_news, 2)
        s = F.softplus(out[:, :, 0:1])
        lam = 0.001 + 0.499 * torch.sigmoid(out[:, :, 1:2])
        dt = (bt.unsqueeze(1) - nt).float().clamp(min=0) / 60.0
        return (s * torch.exp(-lam * dt.unsqueeze(2)) * mk.unsqueeze(2).float()).sum(1)

    def forward_ts_only(self, tf):
        return self.ts_head(self.ts_net(tf))

    def forward(self, tf, nf, nt, bt, mk):
        base = self.forward_ts_only(tf)
        signal = self._signal(nf, nt, bt, mk)
        return base * torch.exp(self.residual_head(signal).clamp(-2, 2))


# ═════════════════════════════════════════════════════════════════════════════
# Datasets
# ═════════════════════════════════════════════════════════════════════════════
class BarFeatDataset(torch.utils.data.Dataset):
    """For variants 2-5: each bar has a precomputed feature vector."""

    def __init__(self, df, target_col, bar_features):
        self.tf = df[TC].values.astype(np.float32)
        self.tg = df[target_col].values.astype(np.float32).reshape(-1, 1)
        self.bar_feats = bar_features.astype(np.float32)

    def __len__(self): return len(self.tf)

    def __getitem__(self, i):
        return {"tf": self.tf[i], "tg": self.tg[i], "bf": self.bar_feats[i]}


def bar_collate(batch):
    return {
        "tf": torch.tensor(np.stack([b["tf"] for b in batch]), dtype=torch.float32),
        "tg": torch.tensor(np.stack([b["tg"] for b in batch]), dtype=torch.float32),
        "bf": torch.tensor(np.stack([b["bf"] for b in batch]), dtype=torch.float32),
    }


class ArticleDataset(torch.utils.data.Dataset):
    """For variants 6-8: returns per-bar list of (article features, timestamps)
    after applying nonreactive mask."""

    def __init__(self, df, target_col, left, right, feat, news_ts, nonreactive):
        self.tf = df[TC].values.astype(np.float32)
        self.tg = df[target_col].values.astype(np.float32).reshape(-1, 1)
        self.bt = df["timestamp"].values.astype(np.float64)
        self.left = left; self.right = right
        self.feat = feat; self.news_ts = news_ts
        self.nonreactive = nonreactive
        self.feat_dim = feat.shape[1]

    def __len__(self): return len(self.tf)

    def __getitem__(self, i):
        idx = np.arange(self.left[i], self.right[i])
        if len(idx) > 0:
            idx = idx[self.nonreactive[idx]]
        if len(idx) > 0:
            nf = self.feat[idx]; nt = self.news_ts[idx]
        else:
            nf = np.zeros((0, self.feat_dim), dtype=np.float32)
            nt = np.zeros((0,), dtype=np.float64)
        return {"tf": self.tf[i], "tg": self.tg[i], "bt": self.bt[i], "nf": nf, "nt": nt}


def article_collate(feat_dim):
    def collate(batch):
        bsz = len(batch)
        counts = [b["nf"].shape[0] for b in batch]
        max_n = max(max(counts), 1)
        nf = np.zeros((bsz, max_n, feat_dim), dtype=np.float32)
        nt = np.zeros((bsz, max_n), dtype=np.float64)
        mk = np.zeros((bsz, max_n), dtype=bool)
        for i, b in enumerate(batch):
            count = b["nf"].shape[0]
            if count > 0:
                nf[i, :count] = b["nf"]; nt[i, :count] = b["nt"]; mk[i, :count] = True
        return {
            "tf": torch.tensor(np.stack([b["tf"] for b in batch]), dtype=torch.float32),
            "tg": torch.tensor(np.stack([b["tg"] for b in batch]), dtype=torch.float32),
            "bt": torch.tensor(np.array([b["bt"] for b in batch]), dtype=torch.float64),
            "nf": torch.tensor(nf, dtype=torch.float32),
            "nt": torch.tensor(nt, dtype=torch.float64),
            "mk": torch.tensor(mk),
        }
    return collate


# ═════════════════════════════════════════════════════════════════════════════
# Per-bar feature constructors (for variants 2-5)
# ═════════════════════════════════════════════════════════════════════════════
def build_bar_features(variant, bars, news_timestamps, nonreactive_mask, sentiment_probs,
                       cryptobert_pca=None):
    """Returns (bar_features ndarray of shape (n_bars, feat_dim), feat_dim)."""
    bar_ts = bars["timestamp"].values.astype(np.int64)
    n_bars = len(bar_ts)
    nts = news_timestamps.astype(np.int64)
    nr = nonreactive_mask.astype(bool)

    if variant == "daily_sentiment":
        # Mean sentiment of articles published today (00:00 UTC up to bar time)
        bars_dt = pd.to_datetime(bar_ts, unit="s", utc=True)
        day_starts = bars_dt.normalize().astype("int64") // 10**9
        feats = np.zeros((n_bars, 3), dtype=np.float32)
        # Use searchsorted to find article ranges per bar
        left = np.searchsorted(nts, day_starts, side="left")
        right = np.searchsorted(nts, bar_ts, side="left")
        for i in range(n_bars):
            idx = np.arange(left[i], right[i])
            idx = idx[nr[idx]]
            if len(idx) > 0:
                feats[i] = sentiment_probs[idx].mean(axis=0)
        return feats, 3

    if variant == "news_count":
        # Count of non-reactive articles in 12hr lookback
        left = np.searchsorted(nts, bar_ts - LOOKBACK_SECONDS, side="left")
        right = np.searchsorted(nts, bar_ts, side="left")
        counts = np.zeros((n_bars, 1), dtype=np.float32)
        for i in range(n_bars):
            idx = np.arange(left[i], right[i])
            counts[i, 0] = nr[idx].sum() if len(idx) else 0.0
        return counts, 1

    if variant == "sentiment_recency":
        # Recency-weighted sentiment over 12hr lookback. Weight = 1/(1 + age_minutes).
        left = np.searchsorted(nts, bar_ts - LOOKBACK_SECONDS, side="left")
        right = np.searchsorted(nts, bar_ts, side="left")
        feats = np.zeros((n_bars, 3), dtype=np.float32)
        for i in range(n_bars):
            idx = np.arange(left[i], right[i])
            idx = idx[nr[idx]]
            if len(idx) == 0:
                continue
            age_min = (bar_ts[i] - nts[idx]) / 60.0
            w = 1.0 / (1.0 + age_min)
            w_sum = w.sum()
            if w_sum > 0:
                feats[i] = (sentiment_probs[idx] * w[:, None]).sum(axis=0) / w_sum
        return feats, 3

    if variant == "cryptobert_pca256_mean":
        assert cryptobert_pca is not None
        K = cryptobert_pca.shape[1]
        left = np.searchsorted(nts, bar_ts - LOOKBACK_SECONDS, side="left")
        right = np.searchsorted(nts, bar_ts, side="left")
        feats = np.zeros((n_bars, K), dtype=np.float32)
        for i in range(n_bars):
            idx = np.arange(left[i], right[i])
            idx = idx[nr[idx]]
            if len(idx) > 0:
                feats[i] = cryptobert_pca[idx].mean(axis=0)
        return feats, K

    raise ValueError(f"Unknown bar-feat variant: {variant}")


# ═════════════════════════════════════════════════════════════════════════════
# Training routines
# ═════════════════════════════════════════════════════════════════════════════
def train_p1(loaders, device, lr=P1_LR, max_epochs=100, patience=15):
    set_seed(P1_SEED)
    model = TSOnlyModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best, pat, best_sd = float("inf"), 0, None
    for ep in range(max_epochs):
        model.train()
        for batch in loaders["train"]:
            tf = batch["tf"].to(device); tgt = batch["tg"].to(device)
            loss = level_loss(model(tf), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in loaders["val"]:
                tf = batch["tf"].to(device); tgt = batch["tg"].to(device)
                vl += level_loss(model(tf), tgt).item() * tf.size(0); vn += tf.size(0)
        vl /= max(vn, 1)
        if vl < best:
            best, pat = vl, 0
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if pat >= patience: break
    model.load_state_dict(best_sd)
    return clone_ts_state(model), float(best), best_sd


def train_p2_bar(model, loaders, device, lr=P2_LR, max_epochs=P2_MAX_EPOCHS,
                  patience=P2_PATIENCE, huber_delta=0.0):
    """P2 for bar-feature variants: freeze TS, train news_head."""
    if huber_delta > 0:
        def loss_fn(p, t): return huber_log_loss(p, t, delta=huber_delta)
    else:
        loss_fn = level_loss
    for p in model.ts_net.parameters(): p.requires_grad = False
    for p in model.ts_head.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    best, pat, best_sd = float("inf"), 0, None
    for ep in range(max_epochs):
        model.train()
        for batch in loaders["train"]:
            tf = batch["tf"].to(device); tgt = batch["tg"].to(device); bf = batch["bf"].to(device)
            loss = loss_fn(model(tf, bf), tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in loaders["val"]:
                tf = batch["tf"].to(device); tgt = batch["tg"].to(device); bf = batch["bf"].to(device)
                vl += loss_fn(model(tf, bf), tgt).item() * tf.size(0); vn += tf.size(0)
        vl /= max(vn, 1)
        if vl < best:
            best, pat = vl, 0
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if pat >= patience: break
    model.load_state_dict(best_sd)
    return model, float(best)


def train_p2_article(model, loaders, device, lr=P2_LR, max_epochs=P2_MAX_EPOCHS,
                      patience=P2_PATIENCE, huber_delta=0.0):
    """P2 for per-article decay variants: freeze TS, train news_head + residual."""
    if huber_delta > 0:
        def loss_fn(p, t): return huber_log_loss(p, t, delta=huber_delta)
    else:
        loss_fn = level_loss
    for p in model.ts_net.parameters(): p.requires_grad = False
    for p in model.ts_head.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    best, pat, best_sd = float("inf"), 0, None
    for ep in range(max_epochs):
        model.train()
        for batch in loaders["train"]:
            tf = batch["tf"].to(device); tgt = batch["tg"].to(device)
            nf = batch["nf"].to(device); nt = batch["nt"].to(device)
            bt = batch["bt"].to(device); mk = batch["mk"].to(device)
            loss = loss_fn(model(tf, nf, nt, bt, mk), tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in loaders["val"]:
                tf = batch["tf"].to(device); tgt = batch["tg"].to(device)
                nf = batch["nf"].to(device); nt = batch["nt"].to(device)
                bt = batch["bt"].to(device); mk = batch["mk"].to(device)
                vl += loss_fn(model(tf, nf, nt, bt, mk), tgt).item() * tf.size(0); vn += tf.size(0)
        vl /= max(vn, 1)
        if vl < best:
            best, pat = vl, 0
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if pat >= patience: break
    model.load_state_dict(best_sd)
    return model, float(best)


def predict_bar(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            tf = batch["tf"].to(device); bf = batch["bf"].to(device)
            preds.append(model(tf, bf).cpu().numpy())
    return np.concatenate(preds).squeeze(1)


def predict_article(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            tf = batch["tf"].to(device); nf = batch["nf"].to(device)
            nt = batch["nt"].to(device); bt = batch["bt"].to(device)
            mk = batch["mk"].to(device)
            preds.append(model(tf, nf, nt, bt, mk).cpu().numpy())
    return np.concatenate(preds).squeeze(1)


def predict_ts_only(p1_state, loader, device, has_news=False):
    model = TSOnlyModel().to(device)
    sd = model.state_dict()
    for k, v in p1_state.items():
        if k in sd:
            sd[k] = v.clone()
    model.load_state_dict(sd)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            tf = batch["tf"].to(device)
            preds.append(model(tf).cpu().numpy())
    return np.concatenate(preds).squeeze(1)


# ═════════════════════════════════════════════════════════════════════════════
# Driver
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", nargs="+", type=int, default=list(range(1, len(WINDOWS) + 1)))
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Subset of variants to run. Defaults to all 8.")
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--huber-delta", type=float, default=DEFAULT_HUBER_DELTA,
                        help="Huber-on-log(vol) delta. 0 = MSE. Default 0.5 (matches v3).")
    args = parser.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Device: {DEVICE}; horizon: {HORIZON}; out: {out_dir}")
    log(f"Loss: {'Huber-on-log delta=' + str(args.huber_delta) if args.huber_delta > 0 else 'MSE-on-log'}; "
        f"P2 LR={P2_LR:g}, patience={P2_PATIENCE}")

    bars = build_price_bars("BTC")
    news = load_news_features("BTC")
    bars, left, right = align_bars_with_news(bars, news["timestamps"])
    log(f"Bars after news overlap: {len(bars):,}")
    log(f"Non-reactive articles: {int(news['nonreactive_mask'].sum()):,} of {len(news['timestamps']):,}")

    # Load CryptoBERT raw (768-d) and pre-fit a global PCA-128 feature for variant 5
    cb = np.load(CRYPTOBERT_PATH)
    cb_si = np.argsort(cb["timestamps"])
    cb_emb = cb["embeddings"][cb_si].astype(np.float32)
    if not np.array_equal(cb["timestamps"][cb_si], news["timestamps"]):
        raise ValueError("CryptoBERT timestamps don't align with news timestamps.")

    # FinBERT CLS embeddings (768-d) for the swap variant
    fb_emb = None
    if FINBERT_CLS_PATH.exists():
        fb = np.load(FINBERT_CLS_PATH)
        fb_si = np.argsort(fb["timestamps"])
        fb_emb = fb["embeddings"][fb_si].astype(np.float32)
        if not np.array_equal(fb["timestamps"][fb_si], news["timestamps"]):
            raise ValueError("FinBERT-CLS timestamps don't align with news timestamps.")
        log(f"FinBERT CLS embeddings: {fb_emb.shape}")

    # CryptoBERT sentiment probs (3-d) for the swap variant
    cb_sent = None
    if CRYPTOBERT_SENT_PATH.exists():
        cb_sent_df = pd.read_parquet(CRYPTOBERT_SENT_PATH).sort_values("unix_timestamp").reset_index(drop=True)
        if not np.array_equal(cb_sent_df["unix_timestamp"].values.astype(np.int64), news["timestamps"]):
            raise ValueError("CryptoBERT sentiment timestamps don't align with news timestamps.")
        cb_sent = cb_sent_df[["p_bullish", "p_neutral", "p_bearish"]].values.astype(np.float32)
        log(f"CryptoBERT sentiment probs: {cb_sent.shape}")

    all_variants = [
        "ts_only",
        "daily_sentiment",
        "news_count",
        "sentiment_recency",
        "cryptobert_pca256_mean",
        "sentiment_decay",
        "cryptobert768_decay",
        "cryptobert_pca256_decay",
        "finbert_cls_decay",          # swap: FinBERT 768-d embeddings
        "cryptobert_sentiment_decay",  # swap: CryptoBERT 3-d sentiment probs
    ]
    variants = args.variants or all_variants
    log(f"Variants: {variants}")

    rows = []
    bars_dt = bars["datetime"]
    pred_store = {}  # window -> {variant: pred_array, "actual": actual}

    for w_num in args.windows:
        if w_num > len(WINDOWS): continue
        val_start, val_end, test_start, test_end = WINDOWS[w_num - 1]
        vs = pd.Timestamp(val_start, tz="UTC"); ve = pd.Timestamp(val_end, tz="UTC")
        ts = pd.Timestamp(test_start, tz="UTC"); te = pd.Timestamp(test_end, tz="UTC")
        train_idx = np.where(bars_dt < vs)[0]
        val_idx = np.where((bars_dt >= vs) & (bars_dt <= ve))[0]
        test_idx = np.where((bars_dt >= ts) & (bars_dt <= te))[0]
        if len(train_idx) < 1000 or len(val_idx) < 100 or len(test_idx) < 100:
            log(f"Window {w_num}: skipped"); continue

        log(""); log("=" * 72)
        log(f"Window {w_num}: test {test_start} -> {test_end}  "
            f"Train={len(train_idx):,} Val={len(val_idx):,} Test={len(test_idx):,}")
        log("=" * 72)

        bars_norm, _, _ = normalize_bars(bars, train_idx)
        actual = bars[TARGET_COL].iloc[test_idx].to_numpy(dtype=np.float64)

        # Per-window PCAs of CryptoBERT (fit on train articles only)
        train_news_mask = (news["timestamps"] >= bars["timestamp"].iloc[train_idx[0]]) & \
                          (news["timestamps"] <= bars["timestamp"].iloc[train_idx[-1]])
        train_emb = cb_emb[train_news_mask]

        # Build a TS-only loader once per window (used by P1 + ts_only variant)
        ts_train_loader = torch.utils.data.DataLoader(
            BarFeatDataset(bars_norm.iloc[train_idx].reset_index(drop=True), TARGET_COL,
                           np.zeros((len(train_idx), 1), dtype=np.float32)),
            batch_size=BATCH_SIZE, shuffle=True, collate_fn=bar_collate, num_workers=0)
        ts_val_loader = torch.utils.data.DataLoader(
            BarFeatDataset(bars_norm.iloc[val_idx].reset_index(drop=True), TARGET_COL,
                           np.zeros((len(val_idx), 1), dtype=np.float32)),
            batch_size=BATCH_SIZE, shuffle=False, collate_fn=bar_collate, num_workers=0)
        ts_test_loader = torch.utils.data.DataLoader(
            BarFeatDataset(bars_norm.iloc[test_idx].reset_index(drop=True), TARGET_COL,
                           np.zeros((len(test_idx), 1), dtype=np.float32)),
            batch_size=BATCH_SIZE, shuffle=False, collate_fn=bar_collate, num_workers=0)

        log(f"  P1 (TS-only)...")
        t0 = time.time()
        ts_state, p1_val, p1_full_sd = train_p1(
            {"train": ts_train_loader, "val": ts_val_loader}, DEVICE)
        log(f"  P1 done [{time.time()-t0:.0f}s] val={p1_val:.4f}")
        torch.save(p1_full_sd, out_dir / f"p1_w{w_num}.pt")

        ts_pred = predict_ts_only(p1_full_sd, ts_test_loader, DEVICE)
        pred_store.setdefault(w_num, {"actual": actual})["ts_only"] = ts_pred

        for variant in variants:
            if variant == "ts_only":
                preds = ts_pred  # already computed
                vl_used = p1_val
                t_var = 0.0
            elif variant in ("daily_sentiment", "news_count", "sentiment_recency", "cryptobert_pca256_mean"):
                # Build per-bar features
                t_feat = time.time()
                if variant == "cryptobert_pca256_mean":
                    pca = PCA(n_components=256, random_state=SEED)
                    pca.fit(train_emb)
                    cb_pca256 = pca.transform(cb_emb).astype(np.float32)
                    feats_full, fdim = build_bar_features(
                        variant, bars, news["timestamps"], news["nonreactive_mask"],
                        news["sentiment_probs"], cryptobert_pca=cb_pca256)
                else:
                    feats_full, fdim = build_bar_features(
                        variant, bars, news["timestamps"], news["nonreactive_mask"],
                        news["sentiment_probs"])
                feat_train = feats_full[train_idx]; feat_val = feats_full[val_idx]; feat_test = feats_full[test_idx]
                # Normalize bar features using train mean/std (skip for news_count which is small int)
                if variant != "news_count":
                    fm = feat_train.mean(0); fs = feat_train.std(0) + 1e-8
                    feat_train = (feat_train - fm) / fs
                    feat_val = (feat_val - fm) / fs
                    feat_test = (feat_test - fm) / fs
                else:
                    feat_train = np.log1p(feat_train); feat_val = np.log1p(feat_val); feat_test = np.log1p(feat_test)

                ds_t = BarFeatDataset(bars_norm.iloc[train_idx].reset_index(drop=True), TARGET_COL, feat_train)
                ds_v = BarFeatDataset(bars_norm.iloc[val_idx].reset_index(drop=True), TARGET_COL, feat_val)
                ds_te = BarFeatDataset(bars_norm.iloc[test_idx].reset_index(drop=True), TARGET_COL, feat_test)
                tr = torch.utils.data.DataLoader(ds_t, batch_size=BATCH_SIZE, shuffle=True, collate_fn=bar_collate, num_workers=0)
                va = torch.utils.data.DataLoader(ds_v, batch_size=BATCH_SIZE, shuffle=False, collate_fn=bar_collate, num_workers=0)
                tee = torch.utils.data.DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, collate_fn=bar_collate, num_workers=0)
                t_feat = time.time() - t_feat

                set_seed(SEED)
                m = BarFeatureResidual(feat_dim=fdim, hidden=16).to(DEVICE)
                load_ts_state(m, ts_state)
                t0 = time.time()
                m, vl_used = train_p2_bar(m, {"train": tr, "val": va}, DEVICE,
                                            huber_delta=args.huber_delta)
                preds = predict_bar(m, tee, DEVICE)
                t_var = time.time() - t0
                log(f"  {variant}: feat_dim={fdim}, P2 val={vl_used:.4f} [feat {t_feat:.0f}s + train {t_var:.0f}s]")
            else:
                # Per-article decay variants
                if variant == "sentiment_decay":
                    feat = news["sentiment_probs"]; fdim = 3; norm_input = False
                elif variant == "cryptobert768_decay":
                    feat = cb_emb; fdim = cb_emb.shape[1]; norm_input = True
                elif variant == "cryptobert_pca256_decay":
                    pca = PCA(n_components=256, random_state=SEED)
                    pca.fit(train_emb)
                    feat = pca.transform(cb_emb).astype(np.float32); fdim = 256; norm_input = True
                elif variant == "finbert_cls_decay":
                    if fb_emb is None:
                        raise FileNotFoundError(f"{FINBERT_CLS_PATH} not found")
                    feat = fb_emb; fdim = fb_emb.shape[1]; norm_input = True
                elif variant == "cryptobert_sentiment_decay":
                    if cb_sent is None:
                        raise FileNotFoundError(f"{CRYPTOBERT_SENT_PATH} not found")
                    feat = cb_sent; fdim = 3; norm_input = False
                else:
                    raise ValueError(f"Unknown variant: {variant}")

                ds_t = ArticleDataset(bars_norm.iloc[train_idx].reset_index(drop=True), TARGET_COL,
                                       left[train_idx], right[train_idx], feat, news["timestamps"], news["nonreactive_mask"])
                ds_v = ArticleDataset(bars_norm.iloc[val_idx].reset_index(drop=True), TARGET_COL,
                                       left[val_idx], right[val_idx], feat, news["timestamps"], news["nonreactive_mask"])
                ds_te = ArticleDataset(bars_norm.iloc[test_idx].reset_index(drop=True), TARGET_COL,
                                       left[test_idx], right[test_idx], feat, news["timestamps"], news["nonreactive_mask"])
                tr = torch.utils.data.DataLoader(ds_t, batch_size=BATCH_SIZE, shuffle=True, collate_fn=article_collate(fdim), num_workers=0)
                va = torch.utils.data.DataLoader(ds_v, batch_size=BATCH_SIZE, shuffle=False, collate_fn=article_collate(fdim), num_workers=0)
                tee = torch.utils.data.DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, collate_fn=article_collate(fdim), num_workers=0)

                set_seed(SEED)
                m = PerArticleDecayResidual(feat_dim=fdim, hidden=16, normalize_input=norm_input).to(DEVICE)
                load_ts_state(m, ts_state)
                t0 = time.time()
                m, vl_used = train_p2_article(m, {"train": tr, "val": va}, DEVICE,
                                                huber_delta=args.huber_delta)
                preds = predict_article(m, tee, DEVICE)
                t_var = time.time() - t0
                log(f"  {variant}: feat_dim={fdim}, P2 val={vl_used:.4f} [{t_var:.0f}s]")

            pred_store[w_num][variant] = preds
            metrics = compute_metrics(preds, actual)
            rows.append({
                "window": w_num, "test_period": f"{test_start}->{test_end}",
                "variant": variant, "p1_val": p1_val, "p2_val": vl_used,
                "MSE": metrics["MSE"], "QLIKE": metrics["QLIKE"],
                "R2": metrics["R2"], "DA": metrics["DA"], "N": len(actual),
                "elapsed_s": t_var,
            })
            # Save predictions per window+variant
            np.savez(out_dir / f"window_{w_num}_predictions.npz",
                     **{v: pred_store[w_num].get(v) for v in pred_store[w_num] if pred_store[w_num].get(v) is not None})

        # Save metrics after each window
        pd.DataFrame(rows).to_csv(out_dir / "expanding_metrics.csv", index=False, float_format="%.10f")

    # Pooled summary
    df = pd.DataFrame(rows)
    pooled = (df.groupby("variant")
                .agg(R2_mean=("R2", "mean"), R2_std=("R2", "std"),
                     MSE_mean=("MSE", "mean"), QLIKE_mean=("QLIKE", "mean"),
                     DA_mean=("DA", "mean"),
                     n_windows=("window", "nunique"))
                .reset_index().sort_values("R2_mean", ascending=False))
    pooled.to_csv(out_dir / "pooled_summary.csv", index=False, float_format="%.10f")
    log("\n=== Pooled summary (sorted by R2 mean) ===")
    log(pooled.to_string(index=False))

    # DM tests vs ts_only on pooled predictions
    if "ts_only" in df["variant"].unique() and len(args.windows) >= 2:
        log("\n=== Pooled DM tests vs ts_only ===")
        ws = sorted(pred_store.keys())
        actual_pool = np.concatenate([pred_store[w]["actual"] for w in ws])
        ts_pool = np.concatenate([pred_store[w]["ts_only"] for w in ws])
        ts_metrics = compute_metrics(ts_pool, actual_pool)
        dm_rows = []
        for v in df["variant"].unique():
            if v == "ts_only": continue
            try:
                pool = np.concatenate([pred_store[w][v] for w in ws])
            except KeyError:
                continue
            m = compute_metrics(pool, actual_pool)
            t_mse, p_mse = dm_test(m["mse_vec"], ts_metrics["mse_vec"], 16)
            t_q, p_q = dm_test(m["qlike_vec"], ts_metrics["qlike_vec"], 16)
            dm_rows.append({"variant": v, "R2": m["R2"], "MSE": m["MSE"], "QLIKE": m["QLIKE"],
                            "DM_MSE_t": t_mse, "DM_MSE_p": p_mse,
                            "DM_QLIKE_t": t_q, "DM_QLIKE_p": p_q})
        dm_df = pd.DataFrame(dm_rows).sort_values("R2", ascending=False)
        dm_df.to_csv(out_dir / "dm_vs_ts_only.csv", index=False, float_format="%.10f")
        log(dm_df.to_string(index=False))


if __name__ == "__main__":
    main()
