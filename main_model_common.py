import argparse
import json
import os
import re
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sp_stats
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset

from step3_topic_flags import TOPIC_RULES, article_score, compile_rules, primary_from_scores

LOG_EPS = 1e-10
LOOKBACK_SECONDS = 12 * 3600
HORIZONS = OrderedDict([
    ("1h", {"target": "RV_1h", "bars": 4, "dm_bandwidth": 4}),
    ("4h", {"target": "RV_4h", "bars": 16, "dm_bandwidth": 16}),
    ("8h", {"target": "RV_8h", "bars": 32, "dm_bandwidth": 32}),
])
SEED = 42
P1_SEED = 7
AUX_WEIGHT = 0.1
LAMBDA_INIT_HALF_LIFE_MIN = 120.0

LR_P1 = 1.5e-3
LR_P2 = 5e-4
LR_EGARCH_NED = 1.5e-4

TC = [
    "rv_4", "rv_16", "rv_96", "rv_384", "log_ret", "momentum_4",
    "momentum_16", "hl_range_4", "volume_ratio_4", "ret_skew_16",
    "ret_kurt_16", "log_close",
]
NT = len(TC)
HAR_FEATURES = ["rv_4", "rv_16", "rv_96", "rv_384"]

WINDOWS = [
    ("2021-07-01", "2021-12-31", "2022-01-01", "2022-06-30"),
    ("2022-01-01", "2022-06-30", "2022-07-01", "2022-12-31"),
    ("2022-07-01", "2022-12-31", "2023-01-01", "2023-06-30"),
    ("2023-01-01", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("2023-07-01", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("2024-01-01", "2024-06-30", "2024-07-01", "2024-12-31"),
    ("2024-07-01", "2024-12-31", "2025-01-01", "2025-06-30"),
    ("2025-01-01", "2025-06-30", "2025-07-01", "2025-12-31"),
]

ASSETS = {
    "BTC": {
        "price": "data/price/btcusd_bitstamp_1min_2012-2025.csv.gz",
        "price_files": [
            "data/price/btcusd_bitstamp_1min_2012-2025.csv.gz",
            "data/price/btcusd_bitstamp_1min_latest.csv",
        ],
        # Keep the tiny 2026 lookahead needed to compute end-2025 RV targets.
        "price_end": "2026-01-02",
        "emb": "data/processed/btc_news_embeddings.npz",
        "emb_cryptobert": "data/processed/btc_news_embeddings_cryptobert.npz",
        "sent": "data/processed/btc_news_sentiment.npz",
        "topic": "data/processed/btc_topic_flags.npz",
        "topic_summary": "results/metadata/btc_topic_schema_summary.csv",
        "results": "results/main/btc_sentiment",
        "terms": r"bitcoin|btc",
    },
    "ETH": {
        "price": "data/price/ETHUSDT_5m_201801_202603.csv",
        "emb": "data/processed/eth_news_embeddings.npz",
        "emb_cryptobert": "data/processed/eth_news_embeddings_cryptobert.npz",
        "sent": "data/processed/eth_news_sentiment.npz",
        "topic": "data/processed/eth_topic_flags.npz",
        "topic_summary": "results/metadata/eth_topic_schema_summary.csv",
        "results": "results/main/eth_sentiment",
        "terms": r"ether(?:eum)?|eth",
    },
}


def log(msg):
    print(msg, flush=True)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def lambda_raw_from_half_life(half_life_min=LAMBDA_INIT_HALF_LIFE_MIN):
    lam = np.log(2.0) / float(half_life_min)
    p = (lam - 0.001) / 0.499
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"Half-life {half_life_min} min maps outside lambda bounds")
    return float(np.log(p / (1.0 - p)))


def init_lambda_raw_(topic_lambda_raw, other_lambda_raw=None, half_life_min=LAMBDA_INIT_HALF_LIFE_MIN):
    raw = lambda_raw_from_half_life(half_life_min)
    with torch.no_grad():
        topic_lambda_raw.fill_(raw)
        if other_lambda_raw is not None:
            other_lambda_raw.fill_(raw)
    return raw


def cfg(asset):
    asset = asset.upper()
    if asset not in ASSETS:
        raise ValueError(f"Unsupported asset: {asset}")
    return ASSETS[asset]


def default_results_dir(asset, news_asset=None):
    asset = asset.upper()
    news_asset = (news_asset or asset).upper()
    if news_asset == asset:
        return cfg(asset)["results"]
    return f"results/main/{asset.lower()}_{news_asset.lower()}_news"


def get_horizon_cfg(horizon):
    if horizon not in HORIZONS:
        raise ValueError(f"Unsupported horizon: {horizon}")
    return HORIZONS[horizon]


# Tags from the news feed that indicate market-recap / price-action articles.
# These are price/TA/recurring-column publications that describe past price
# moves rather than carrying causal information; the model should not see them.
RECAP_TAGS = {
    "price-news",
    "Markets News",
    "Market Wrap",
    "Bitcoin Price",
    "Technical Analysis",
    "First Mover",
    "Bitcoin Long & Short",
    "Crypto Long & Short",
    "Markets Daily",
    "Asia Pacific",
    "Trading",
}

# A small high-precision regex augmentation for articles that the publisher
# didn't tag as recap but are obviously price-recap by title (catches a few
# hundred more articles). Patterns are intentionally narrow to avoid the
# over-catching that the original full-regex filter caused on macro/mining news.
SAFE_RECAP_TITLE_PATTERNS = [
    r"(?i)\bmarket\s+wrap\b",
    r"(?i)^(?:bitcoin|btc)\s+\$",
    r"(?i)^first\s+mover\s+(asia|americas|europe)",
]


def reactive_patterns(asset):  # kept for backward compatibility; no longer the primary filter
    return SAFE_RECAP_TITLE_PATTERNS


def tag_based_recap_mask(news_df):
    """Return a boolean array (len = len(news_df)) marking articles whose
    `tags` column contains any tag in RECAP_TAGS, OR whose title matches
    one of the high-precision SAFE_RECAP_TITLE_PATTERNS."""
    tags_col = news_df["tags"].fillna("").astype(str)
    tagged = tags_col.apply(
        lambda t: any(tag in [p.strip() for p in t.split("|")] for tag in RECAP_TAGS)
    )
    titles = news_df["title"].fillna("").astype(str)
    title_pat = re.compile("|".join(SAFE_RECAP_TITLE_PATTERNS))
    title_hits = titles.str.contains(title_pat, regex=True, na=False)
    return (tagged | title_hits).values


def build_price_bars(asset):
    c = cfg(asset)
    price_paths = c.get("price_files") or [c["price"]]
    frames = []
    for path in price_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{asset.upper()} price file not found: {path}")
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    if c.get("price_end"):
        price_end = pd.Timestamp(c["price_end"], tz="UTC")
        df = df[df["datetime"] <= price_end]
    df = df.sort_values("datetime").reset_index(drop=True).set_index("datetime")
    bars = df.resample("15min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum", "timestamp": "first",
    }).dropna(subset=["close"]).reset_index()
    bars["log_ret"] = np.log(bars["close"] / bars["close"].shift(1))
    bars["log_ret_sq"] = bars["log_ret"] ** 2
    sq = pd.Series(bars["log_ret_sq"].values).shift(-1)
    bars["RV_1h"] = sq.iloc[::-1].rolling(4, min_periods=4).sum().iloc[::-1].values
    bars["RV_4h"] = sq.iloc[::-1].rolling(16, min_periods=16).sum().iloc[::-1].values
    bars["RV_8h"] = sq.iloc[::-1].rolling(32, min_periods=32).sum().iloc[::-1].values
    bars["rv_4"] = bars["log_ret_sq"].rolling(4).sum()
    bars["rv_16"] = bars["log_ret_sq"].rolling(16).sum()
    bars["rv_96"] = bars["log_ret_sq"].rolling(96).sum()
    bars["rv_384"] = bars["log_ret_sq"].rolling(384).sum()
    bars["momentum_4"] = bars["log_ret"].rolling(4).sum()
    bars["momentum_16"] = bars["log_ret"].rolling(16).sum()
    bars["hl_range_4"] = ((bars["high"] - bars["low"]) / bars["close"]).rolling(4).mean()
    bars["volume_ratio_4"] = bars["volume"] / bars["volume"].rolling(4).mean().clip(lower=1e-10)
    bars["ret_skew_16"] = bars["log_ret"].rolling(16).apply(lambda x: skew(x, nan_policy="omit"), raw=True)
    bars["ret_kurt_16"] = bars["log_ret"].rolling(16).apply(lambda x: kurtosis(x, nan_policy="omit"), raw=True)
    bars["log_close"] = np.log(bars["close"])
    return bars.dropna(subset=TC + ["RV_1h", "RV_4h", "RV_8h"]).reset_index(drop=True)


def build_topic_file(asset, news_asset):
    c = cfg(asset)
    rules = compile_rules()
    topic_names = np.array(list(TOPIC_RULES.keys()), dtype=object)
    scores = np.stack([article_score(row, rules) for _, row in news_asset.iterrows()])
    flags = (scores > 0).astype(np.uint8)
    primary = np.array([primary_from_scores(row, topic_names) for row in scores], dtype=object)
    match_count = flags.sum(axis=1).astype(np.int16)
    # Tag-based recap filter (Option C): drop articles tagged as price/markets
    # recap or matching the small high-precision title regex.
    recap_mask = tag_based_recap_mask(news_asset)
    nonreactive = (~recap_mask).astype(np.uint8)
    os.makedirs(os.path.dirname(c["topic"]), exist_ok=True)
    np.savez(
        c["topic"], topic_names=topic_names, topic_scores=scores.astype(np.int16),
        topic_flags=flags, primary_topic=primary, match_count=match_count,
        nonreactive_mask=nonreactive,
    )
    rows = []
    nr = max(int(nonreactive.sum()), 1)
    for i, topic in enumerate(topic_names):
        rows.append({
            "asset": asset.upper(), "topic": topic,
            "count_all": int(flags[:, i].sum()),
            "share_all": float(flags[:, i].mean()),
            "count_nonreactive": int(flags[nonreactive.astype(bool), i].sum()),
            "share_nonreactive": int(flags[nonreactive.astype(bool), i].sum()) / nr,
        })
    os.makedirs(os.path.dirname(c["topic_summary"]), exist_ok=True)
    pd.DataFrame(rows).to_csv(c["topic_summary"], index=False)


def load_news_features(asset, rebuild_topics=False):
    c = cfg(asset)
    news_df = pd.read_parquet("data/news/processed_news.parquet")
    news_asset = news_df[news_df["asset"] == asset.upper()].reset_index(drop=True)
    if rebuild_topics or not os.path.exists(c["topic"]):
        log(f"Building {asset.upper()} topic/category flags...")
        build_topic_file(asset, news_asset)
    emb = np.load(c["emb"])
    sent = np.load(c["sent"])["sentiment_probs"].astype(np.float32)
    topic = np.load(c["topic"], allow_pickle=True)
    n = len(emb["timestamps"])
    if len(news_asset) != n or len(sent) != n or len(topic["topic_flags"]) != n:
        raise ValueError(
            f"{asset.upper()} news feature length mismatch: "
            f"news={len(news_asset)}, embeddings={n}, sentiment={len(sent)}, "
            f"topics={len(topic['topic_flags'])}"
        )
    si = np.argsort(emb["timestamps"])
    nts = emb["timestamps"][si]
    # Tag-based recap mask: keep aligned with news_asset's row order, then
    # reorder by the embedding sort index to match the rest of the features.
    recap_mask_news_order = tag_based_recap_mask(news_asset)
    # news_asset is in original row order; emb timestamps were sorted by si.
    # We need a mapping from news_asset row -> emb row. Both should be aligned
    # by unix_timestamp; sort both by timestamp consistently.
    news_ts_orig = news_asset["unix_timestamp"].values.astype(np.int64)
    news_si = np.argsort(news_ts_orig)
    recap_mask_sorted = recap_mask_news_order[news_si]
    if not np.array_equal(news_ts_orig[news_si], nts):
        raise ValueError("News timestamps don't align with embedding timestamps after sort")
    nonreactive = ~recap_mask_sorted
    topic_flags = topic["topic_flags"][si].astype(np.float32)
    sent = sent[si]
    return {
        "timestamps": nts,
        "nonreactive_mask": nonreactive,
        "embeddings": emb["embeddings"][si].astype(np.float32),
        "sentiment_probs": sent,
        "topic_flags": topic_flags,
        "topic_names": [str(x) for x in topic["topic_names"].tolist()],
        "feat_topic": np.column_stack([sent, topic_flags]).astype(np.float32),
    }


def align_bars_with_news(bars, news_ts):
    ts = bars["timestamp"].values
    bars = bars[(ts >= max(ts.min(), news_ts.min())) & (ts <= min(ts.max(), news_ts.max()))].reset_index(drop=True)
    ts = bars["timestamp"].values
    left = np.searchsorted(news_ts, ts - LOOKBACK_SECONDS, side="left")
    right = np.searchsorted(news_ts, ts, side="left")
    return bars, left, right


def normalize_bars(bars, train_idx):
    mean = bars[TC].iloc[train_idx].mean().values.astype(np.float32)
    std = bars[TC].iloc[train_idx].std().values.astype(np.float32)
    std[std < 1e-10] = 1.0
    out = bars.copy()
    for i, col in enumerate(TC):
        out[col] = ((out[col] - mean[i]) / std[i]).clip(-10, 10)
    return out, mean, std


class NewsDataset(Dataset):
    def __init__(self, df, target_col, left, right, feat, topics, news_ts, nonreactive):
        self.tf = df[TC].values.astype(np.float32)
        self.tg = df[target_col].values.astype(np.float32).reshape(-1, 1)
        self.bt = df["timestamp"].values.astype(np.float64)
        self.left = left
        self.right = right
        self.feat = feat
        self.topics = topics
        self.news_ts = news_ts
        self.nonreactive = nonreactive
        self.feat_dim = feat.shape[1]
        self.n_topics = topics.shape[1]

    def __len__(self):
        return len(self.tf)

    def __getitem__(self, i):
        idx = np.arange(self.left[i], self.right[i])
        if len(idx) > 0:
            idx = idx[self.nonreactive[idx]]
        if len(idx) > 0:
            nf = self.feat[idx]
            nt = self.news_ts[idx]
            tm = self.topics[idx]
        else:
            nf = np.zeros((0, self.feat_dim), dtype=np.float32)
            nt = np.zeros((0,), dtype=np.float64)
            tm = np.zeros((0, self.n_topics), dtype=np.float32)
        return {"tf": self.tf[i], "tg": self.tg[i], "bt": self.bt[i], "nf": nf, "nt": nt, "tm": tm}


def make_collate(feat_dim, n_topics):
    def collate(batch):
        bsz = len(batch)
        counts = [b["nf"].shape[0] for b in batch]
        max_n = max(max(counts), 1)
        nf = np.zeros((bsz, max_n, feat_dim), dtype=np.float32)
        nt = np.zeros((bsz, max_n), dtype=np.float64)
        tm = np.zeros((bsz, max_n, n_topics), dtype=np.float32)
        mk = np.zeros((bsz, max_n), dtype=bool)
        for i, row in enumerate(batch):
            count = row["nf"].shape[0]
            if count > 0:
                nf[i, :count] = row["nf"]
                nt[i, :count] = row["nt"]
                tm[i, :count] = row["tm"]
                mk[i, :count] = True
        return {
            "tf": torch.tensor(np.stack([row["tf"] for row in batch]), dtype=torch.float32),
            "tg": torch.tensor(np.stack([row["tg"] for row in batch]), dtype=torch.float32),
            "bt": torch.tensor(np.array([row["bt"] for row in batch]), dtype=torch.float64),
            "nf": torch.tensor(nf, dtype=torch.float32),
            "nt": torch.tensor(nt, dtype=torch.float64),
            "tm": torch.tensor(tm, dtype=torch.float32),
            "mk": torch.tensor(mk),
        }
    return collate


def make_loader(df, target_col, left, right, feat, topics, news, batch_size, shuffle):
    ds = NewsDataset(df, target_col, left, right, feat, topics, news["timestamps"], news["nonreactive_mask"])
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        collate_fn=make_collate(feat.shape[1], topics.shape[1]),
        num_workers=0, pin_memory=True,
    )


def make_split_loaders(
    bars_norm, target_col, train_idx, val_idx, test_idx, left, right, feat, topics, news, batch_size,
    train_shuffle=True,
):
    return {
        "train": make_loader(bars_norm.iloc[train_idx].reset_index(drop=True), target_col, left[train_idx], right[train_idx], feat, topics, news, batch_size, train_shuffle),
        "val": make_loader(bars_norm.iloc[val_idx].reset_index(drop=True), target_col, left[val_idx], right[val_idx], feat, topics, news, batch_size, False),
        "test": make_loader(bars_norm.iloc[test_idx].reset_index(drop=True), target_col, left[test_idx], right[test_idx], feat, topics, news, batch_size, False),
    }


class TopicCategoryDecayModel(nn.Module):
    def __init__(self, feat_dim, n_topics, hidden=32):
        super().__init__()
        self.ts_net = nn.Sequential(nn.Linear(NT, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU())
        self.ts_head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Softplus())
        self.s_head = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.topic_lambda_raw = nn.Parameter(torch.zeros(n_topics))
        self.other_lambda_raw = nn.Parameter(torch.zeros(1))
        init_lambda_raw_(self.topic_lambda_raw, self.other_lambda_raw)
        self.residual_head = nn.Linear(1, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.aux_head = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Softplus())

    def lambda_values(self):
        return 0.001 + 0.499 * torch.sigmoid(self.topic_lambda_raw), 0.001 + 0.499 * torch.sigmoid(self.other_lambda_raw)

    def _event_signal(self, nf, nt, bt, mk, tm):
        bsz, n_news, feat_dim = nf.shape
        s = F.softplus(self.s_head(nf.reshape(bsz * n_news, feat_dim)).reshape(bsz, n_news, 1))
        lam_topics, lam_other = self.lambda_values()
        lam_topics = lam_topics.view(1, 1, -1)
        count = tm.sum(dim=2, keepdim=True)
        lam = (tm * lam_topics).sum(dim=2, keepdim=True) / count.clamp(min=1.0)
        lam = torch.where(count > 0, lam, lam_other.view(1, 1, 1))
        dt = (bt.unsqueeze(1) - nt).float().clamp(min=0) / 60.0
        return (s * torch.exp(-lam * dt.unsqueeze(2)) * mk.unsqueeze(2).float()).sum(1)

    def forward_ts_only(self, tf):
        return self.ts_head(self.ts_net(tf))

    def forward(self, tf, nf, nt, bt, mk, tm):
        base = self.forward_ts_only(tf)
        signal = self._event_signal(nf, nt, bt, mk, tm)
        return base * torch.exp(self.residual_head(signal).clamp(-2, 2))

    def forward_aux(self, nf, nt, bt, mk, tm):
        return self.aux_head(self._event_signal(nf, nt, bt, mk, tm))


class EgarchNedModel(nn.Module):
    def __init__(self, feat_dim, n_topics, hidden=32):
        super().__init__()
        self.s_head = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.topic_lambda_raw = nn.Parameter(torch.zeros(n_topics))
        self.other_lambda_raw = nn.Parameter(torch.zeros(1))
        init_lambda_raw_(self.topic_lambda_raw, self.other_lambda_raw)
        self.residual_head = nn.Linear(1, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.aux_head = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Softplus())

    def lambda_values(self):
        return 0.001 + 0.499 * torch.sigmoid(self.topic_lambda_raw), 0.001 + 0.499 * torch.sigmoid(self.other_lambda_raw)

    def _event_signal(self, nf, nt, bt, mk, tm):
        bsz, n_news, feat_dim = nf.shape
        s = F.softplus(self.s_head(nf.reshape(bsz * n_news, feat_dim)).reshape(bsz, n_news, 1))
        lam_topics, lam_other = self.lambda_values()
        lam_topics = lam_topics.view(1, 1, -1)
        count = tm.sum(dim=2, keepdim=True)
        lam = (tm * lam_topics).sum(dim=2, keepdim=True) / count.clamp(min=1.0)
        lam = torch.where(count > 0, lam, lam_other.view(1, 1, 1))
        dt = (bt.unsqueeze(1) - nt).float().clamp(min=0) / 60.0
        return (s * torch.exp(-lam * dt.unsqueeze(2)) * mk.unsqueeze(2).float()).sum(1)

    def forward(self, base, nf, nt, bt, mk, tm):
        signal = self._event_signal(nf, nt, bt, mk, tm)
        return base * torch.exp(self.residual_head(signal).clamp(-2, 2))

    def forward_aux(self, nf, nt, bt, mk, tm):
        return self.aux_head(self._event_signal(nf, nt, bt, mk, tm))


def level_loss(pred, target):
    return F.mse_loss(torch.log(pred + LOG_EPS), torch.log(target + LOG_EPS))


def huber_log_loss(pred, target, delta=0.75):
    """Huber loss on log-vol: quadratic for |err| <= delta, linear beyond.

    Down-weights the tail observations whose squared loss dominates training
    under plain MSE on log(vol).
    """
    return F.huber_loss(
        torch.log(pred + LOG_EPS), torch.log(target + LOG_EPS), delta=delta,
    )


def mse_loss_vec(pred, actual):
    return (np.log(actual + LOG_EPS) - np.log(pred + LOG_EPS)) ** 2


def qlike_loss_vec(pred, actual):
    ratio = (actual + LOG_EPS) / (pred + LOG_EPS)
    return ratio - np.log(ratio) - 1


def da_score(pred, actual):
    med = np.median(actual)
    return ((actual > med) == (pred > med)).mean()


def compute_metrics(pred, actual):
    mse = mse_loss_vec(pred, actual)
    qlike = qlike_loss_vec(pred, actual)
    return {
        "MSE": float(mse.mean()),
        "QLIKE": float(qlike.mean()),
        "R2": float(r2_score(np.log(actual + LOG_EPS), np.log(pred + LOG_EPS))),
        "DA": float(da_score(pred, actual)),
        "mse_vec": mse,
        "qlike_vec": qlike,
    }


def dm_test(loss1, loss2, h):
    d = np.asarray(loss1) - np.asarray(loss2)
    n = len(d)
    d_bar = d.mean()
    gamma_0 = np.mean((d - d_bar) ** 2)
    gamma_sum = 0.0
    for k in range(1, h + 1):
        weight = 1 - k / (h + 1)
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        gamma_sum += 2 * weight * gamma_k
    var_d = (gamma_0 + gamma_sum) / n
    if var_d <= 0:
        return 0.0, 1.0
    t_stat = d_bar / np.sqrt(var_d)
    return float(t_stat), float(2 * sp_stats.norm.sf(abs(t_stat)))


def fit_har(bars, train_idx, test_idx, target_col):
    lr = LinearRegression()
    lr.fit(bars[HAR_FEATURES].iloc[train_idx].values, np.log(bars[target_col].iloc[train_idx].values + LOG_EPS))
    return np.exp(lr.predict(bars[HAR_FEATURES].iloc[test_idx].values))


def fit_egarch_all(bars, train_idx):
    from arch import arch_model
    train_end = train_idx[-1] + 1
    returns = bars["log_ret"].values * 100
    returns[~np.isfinite(returns)] = 0.0
    am = arch_model(returns[:train_end], vol="EGARCH", p=1, o=1, q=1, dist="t", mean="Zero", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    am_all = arch_model(returns, vol="EGARCH", p=1, o=1, q=1, dist="t", mean="Zero", rescale=False)
    filtered = am_all.fix(res.params)
    return np.clip(filtered.conditional_volatility**2 / 10000.0, 1e-15, None)


def clone_ts_state(model):
    return {k: v.cpu().clone() for k, v in model.state_dict().items() if k.startswith("ts_net") or k.startswith("ts_head")}


def load_ts_state(model, ts_state):
    state = model.state_dict()
    for key, value in ts_state.items():
        if key in state:
            state[key] = value.clone()
    model.load_state_dict(state)


def train_main_model(loaders, feat_dim, n_topics, device):
    set_seed(P1_SEED)
    model = TopicCategoryDecayModel(feat_dim, n_topics).to(device)

    for p in model.s_head.parameters():
        p.requires_grad = False
    model.topic_lambda_raw.requires_grad = False
    model.other_lambda_raw.requires_grad = False
    for p in model.residual_head.parameters():
        p.requires_grad = False
    for p in model.aux_head.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR_P1)
    best, pat, p1_sd = float("inf"), 0, None
    for _ in range(100):
        model.train()
        for batch in loaders["train"]:
            tf = batch["tf"].to(device)
            tgt = batch["tg"].to(device)
            loss = level_loss(model.forward_ts_only(tf), tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in loaders["val"]:
                tf = batch["tf"].to(device)
                tgt = batch["tg"].to(device)
                vl += level_loss(model.forward_ts_only(tf), tgt).item() * tf.size(0)
                vn += tf.size(0)
        vl /= max(vn, 1)
        if vl < best:
            best, pat = vl, 0
            p1_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if pat >= 15:
            break
    model.load_state_dict(p1_sd)
    pure_p1_state = clone_ts_state(model)

    for p in model.parameters():
        p.requires_grad = True
    for p in model.ts_net.parameters():
        p.requires_grad = False
    for p in model.ts_head.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR_P2)
    best, pat, p2_sd = float("inf"), 0, None
    for _ in range(120):
        model.train()
        for batch in loaders["train"]:
            tf = batch["tf"].to(device)
            nf = batch["nf"].to(device)
            nt = batch["nt"].to(device)
            bt = batch["bt"].to(device)
            mk = batch["mk"].to(device)
            tm = batch["tm"].to(device)
            tgt = batch["tg"].to(device)
            loss = level_loss(model(tf, nf, nt, bt, mk, tm), tgt)
            loss = loss + AUX_WEIGHT * level_loss(model.forward_aux(nf, nt, bt, mk, tm), tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in loaders["val"]:
                tf = batch["tf"].to(device)
                nf = batch["nf"].to(device)
                nt = batch["nt"].to(device)
                bt = batch["bt"].to(device)
                mk = batch["mk"].to(device)
                tm = batch["tm"].to(device)
                tgt = batch["tg"].to(device)
                vl += level_loss(model(tf, nf, nt, bt, mk, tm), tgt).item() * tf.size(0)
                vn += tf.size(0)
        vl /= max(vn, 1)
        if vl < best:
            best, pat = vl, 0
            p2_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        if pat >= 20:
            break
    model.load_state_dict(p2_sd)
    return model, pure_p1_state


def predict_main_and_ts(model, p1_state, loader, feat_dim, n_topics, device):
    ts_model = TopicCategoryDecayModel(feat_dim, n_topics).to(device)
    load_ts_state(ts_model, p1_state)
    model.eval()
    ts_model.eval()
    ts_preds, main_preds = [], []
    with torch.no_grad():
        for batch in loader:
            tf = batch["tf"].to(device)
            nf = batch["nf"].to(device)
            nt = batch["nt"].to(device)
            bt = batch["bt"].to(device)
            mk = batch["mk"].to(device)
            tm = batch["tm"].to(device)
            ts_preds.append(ts_model.forward_ts_only(tf).cpu().numpy())
            main_preds.append(model(tf, nf, nt, bt, mk, tm).cpu().numpy())
    return np.concatenate(ts_preds).squeeze(1), np.concatenate(main_preds).squeeze(1)


def train_egarch_ned(loaders, egarch_train, egarch_val, egarch_test, device):
    set_seed(SEED)
    model = EgarchNedModel(loaders["train"].dataset.feat_dim, loaders["train"].dataset.n_topics).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_EGARCH_NED)
    train_base = torch.tensor(egarch_train.reshape(-1, 1), dtype=torch.float32)
    val_base = torch.tensor(egarch_val.reshape(-1, 1), dtype=torch.float32)
    best, pat, best_sd = float("inf"), 0, None
    for _ in range(120):
        model.train()
        cursor = 0
        for batch in loaders["train"]:
            n = batch["tg"].shape[0]
            base = train_base[cursor:cursor + n].to(device)
            cursor += n
            nf = batch["nf"].to(device)
            nt = batch["nt"].to(device)
            bt = batch["bt"].to(device)
            mk = batch["mk"].to(device)
            tm = batch["tm"].to(device)
            tgt = batch["tg"].to(device)
            loss = level_loss(model(base, nf, nt, bt, mk, tm), tgt)
            loss = loss + AUX_WEIGHT * level_loss(model.forward_aux(nf, nt, bt, mk, tm), tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        vl, vn, cursor = 0.0, 0, 0
        with torch.no_grad():
            for batch in loaders["val"]:
                n = batch["tg"].shape[0]
                base = val_base[cursor:cursor + n].to(device)
                cursor += n
                nf = batch["nf"].to(device)
                nt = batch["nt"].to(device)
                bt = batch["bt"].to(device)
                mk = batch["mk"].to(device)
                tm = batch["tm"].to(device)
                tgt = batch["tg"].to(device)
                vl += level_loss(model(base, nf, nt, bt, mk, tm), tgt).item() * n
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
            base = test_base[cursor:cursor + n].to(device)
            cursor += n
            nf = batch["nf"].to(device)
            nt = batch["nt"].to(device)
            bt = batch["bt"].to(device)
            mk = batch["mk"].to(device)
            tm = batch["tm"].to(device)
            preds.append(model(base, nf, nt, bt, mk, tm).cpu().numpy())
    return model, np.concatenate(preds).squeeze(1)


def run_expanding_training(
    asset, results_dir=None, batch_size=256, run_dm=True, rebuild_topics=False,
    news_asset=None, horizons=None, windows=None,
):
    asset = asset.upper()
    news_asset = (news_asset or asset).upper()
    horizon_names = list(horizons or HORIZONS.keys())
    window_filter = set(windows or range(1, len(WINDOWS) + 1))
    c = cfg(asset)
    cfg(news_asset)
    results_dir = results_dir or default_results_dir(asset, news_asset)
    for sub in ["checkpoints", "predictions", "metadata"]:
        os.makedirs(os.path.join(results_dir, sub), exist_ok=True)

    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"Price asset: {asset}")
    log(f"News asset: {news_asset}")
    log(f"Results dir: {results_dir}")
    log(f"Horizons: {', '.join(horizon_names)}")
    log(f"Main model LRs: P1={LR_P1:g}, P2={LR_P2:g}")

    bars = build_price_bars(asset)
    news = load_news_features(news_asset, rebuild_topics=rebuild_topics)
    bars, left, right = align_bars_with_news(bars, news["timestamps"])
    log(f"Bars after news overlap: {len(bars):,}")

    rows = []
    lambda_rows = []
    bars_dt = bars["datetime"]

    for window_num, (val_start, val_end, test_start, test_end) in enumerate(WINDOWS, start=1):
        if window_num not in window_filter:
            continue
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
        log("=" * 72)
        log(f"Window {window_num}/{len(WINDOWS)}: test {test_start} -> {test_end}")
        log(f"Train={len(train_idx):,} Val={len(val_idx):,} Test={len(test_idx):,}")
        log("=" * 72)

        bars_norm, norm_mean, norm_std = normalize_bars(bars, train_idx)
        np.savez(
            os.path.join(results_dir, "metadata", f"window_{window_num}_meta.npz"),
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            norm_mean=norm_mean, norm_std=norm_std,
        )

        start = time.time()
        egarch_base_15m = fit_egarch_all(bars, train_idx)
        log(f"EGARCH fit/filter: {time.time() - start:.0f}s")

        for horizon in horizon_names:
            h_cfg = get_horizon_cfg(horizon)
            target_col = h_cfg["target"]
            egarch_all = np.clip(egarch_base_15m * h_cfg["bars"], 1e-15, None)
            log(f"\n  Horizon {horizon} ({target_col})")

            har_pred = fit_har(bars, train_idx, test_idx, target_col)
            main_loaders = make_split_loaders(
                bars_norm, target_col, train_idx, val_idx, test_idx, left, right,
                news["feat_topic"], news["topic_flags"], news, batch_size,
            )
            main_model, p1_state = train_main_model(
                main_loaders, news["feat_topic"].shape[1], news["topic_flags"].shape[1], device,
            )
            lam_topics, lam_other = main_model.lambda_values()
            lam_values = lam_topics.detach().cpu().numpy()
            for topic_name, lam in zip(news["topic_names"], lam_values):
                lambda_rows.append({
                    "asset": asset, "window": window_num, "horizon": horizon,
                    "news_asset": news_asset,
                    "topic": topic_name, "lambda_per_min": float(lam),
                    "half_life_min": float(np.log(2) / lam),
                    "half_life_hours": float(np.log(2) / lam / 60.0),
                })
            other_lam = float(lam_other.detach().cpu().item())
            lambda_rows.append({
                "asset": asset, "window": window_num, "horizon": horizon,
                "news_asset": news_asset,
                "topic": "Other/Unmatched", "lambda_per_min": other_lam,
                "half_life_min": float(np.log(2) / other_lam),
                "half_life_hours": float(np.log(2) / other_lam / 60.0),
            })
            ts_pred, main_pred = predict_main_and_ts(
                main_model, p1_state, main_loaders["test"],
                news["feat_topic"].shape[1], news["topic_flags"].shape[1], device,
            )

            ned_loaders = make_split_loaders(
                bars_norm, target_col, train_idx, val_idx, test_idx, left, right,
                news["feat_topic"], news["topic_flags"], news, batch_size, train_shuffle=False,
            )
            egarch_ned_model, egarch_ned_pred = train_egarch_ned(
                ned_loaders, egarch_all[train_idx], egarch_all[val_idx], egarch_all[test_idx], device,
            )

            actual = bars[target_col].iloc[test_idx].to_numpy(dtype=np.float64)
            preds = OrderedDict([
                ("HAR", har_pred),
                ("EGARCH", egarch_all[test_idx]),
                ("EGARCH+NED", egarch_ned_pred),
                ("TS MLP", ts_pred),
                ("Main Model", main_pred),
            ])
            np.savez(
                os.path.join(results_dir, "predictions", f"window_{window_num}_{horizon}_predictions.npz"),
                actual=actual, har=har_pred, egarch=egarch_all[test_idx],
                egarch_ned=egarch_ned_pred, ts_mlp=ts_pred, main_model=main_pred,
                test_idx=test_idx, datetime=bars["datetime"].iloc[test_idx].astype(str).values,
                horizon=horizon, target_col=target_col, asset=asset, news_asset=news_asset,
            )
            torch.save(main_model.state_dict(), os.path.join(results_dir, "checkpoints", f"main_model_w{window_num}_{horizon}.pt"))
            torch.save(p1_state, os.path.join(results_dir, "checkpoints", f"ts_mlp_w{window_num}_{horizon}.pt"))
            torch.save(egarch_ned_model.state_dict(), os.path.join(results_dir, "checkpoints", f"egarch_ned_w{window_num}_{horizon}.pt"))

            for name, pred in preds.items():
                m = compute_metrics(pred, actual)
                rows.append({
                    "asset": asset, "window": window_num, "test_period": f"{test_start}->{test_end}",
                    "news_asset": news_asset, "horizon": horizon, "model": name, "MSE": m["MSE"], "QLIKE": m["QLIKE"],
                    "R2": m["R2"], "DA": m["DA"], "N": len(actual),
                })
                log(f"    {name:<12s} R2={m['R2']:.4f} MSE={m['MSE']:.5f} QLIKE={m['QLIKE']:.5f} DA={m['DA']:.3f}")

    metrics_path = os.path.join(results_dir, "expanding_window_metrics.csv")
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(metrics_path, index=False, float_format="%.10f")
    mean_path = os.path.join(results_dir, "expanding_window_mean_metrics.csv")
    if not metrics_df.empty:
        mean_df = (
            metrics_df.groupby(["asset", "news_asset", "horizon", "model"], as_index=False)
            .agg(MSE=("MSE", "mean"), QLIKE=("QLIKE", "mean"), R2=("R2", "mean"),
                 DA=("DA", "mean"), N=("N", "sum"), n_windows=("window", "nunique"))
        )
        mean_df.to_csv(mean_path, index=False, float_format="%.10f")
    lambda_path = os.path.join(results_dir, "topic_lambda_summary.csv")
    pd.DataFrame(lambda_rows).to_csv(lambda_path, index=False, float_format="%.10f")
    with open(os.path.join(results_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "asset": asset, "news_asset": news_asset, "horizons": horizon_names,
            "windows": sorted(window_filter), "lr_p1": LR_P1, "lr_p2": LR_P2,
            "models": ["HAR", "EGARCH", "EGARCH+NED", "TS MLP", "Main Model"],
            "topic_names": news["topic_names"],
        }, f, indent=2)
    log(f"Saved: {metrics_path}")
    if not metrics_df.empty:
        log(f"Saved: {mean_path}")
    log(f"Saved: {lambda_path}")
    if run_dm:
        run_dm_evaluation(asset, results_dir, news_asset=news_asset, horizons=horizon_names)


def run_dm_evaluation(asset, results_dir=None, news_asset=None, horizons=None):
    asset = asset.upper()
    news_asset = (news_asset or asset).upper()
    horizon_names = list(horizons or HORIZONS.keys())
    cfg(news_asset)
    results_dir = results_dir or default_results_dir(asset, news_asset)
    pred_dir = os.path.join(results_dir, "predictions")
    model_keys = OrderedDict([
        ("har", "HAR"),
        ("egarch", "EGARCH"),
        ("egarch_ned", "EGARCH+NED"),
        ("ts_mlp", "TS MLP"),
        ("main_model", "Main Model"),
    ])
    rows = []
    for horizon in horizon_names:
        h_cfg = get_horizon_cfg(horizon)
        h_rows = []
        pooled = {key: {"actual": [], "pred": []} for key in model_keys}
        use_legacy_4h_names = (
            horizon == "4h"
            and not any(
                os.path.exists(os.path.join(pred_dir, f"window_{w}_4h_predictions.npz"))
                for w in range(1, len(WINDOWS) + 1)
            )
        )

        for window_num in range(1, len(WINDOWS) + 1):
            path = os.path.join(pred_dir, f"window_{window_num}_{horizon}_predictions.npz")
            if use_legacy_4h_names and not os.path.exists(path):
                path = os.path.join(pred_dir, f"window_{window_num}_predictions.npz")
            if not os.path.exists(path):
                continue
            data = np.load(path, allow_pickle=True)
            actual = data["actual"].astype(np.float64)
            ts_m = compute_metrics(data["ts_mlp"], actual)
            main_m = compute_metrics(data["main_model"], actual)
            for key, name in model_keys.items():
                pred = data[key].astype(np.float64)
                m = compute_metrics(pred, actual)
                if key == "ts_mlp":
                    dm_ts = (np.nan, np.nan, np.nan, np.nan)
                else:
                    dm_ts = (
                        *dm_test(m["mse_vec"], ts_m["mse_vec"], h_cfg["dm_bandwidth"]),
                        *dm_test(m["qlike_vec"], ts_m["qlike_vec"], h_cfg["dm_bandwidth"]),
                    )
                if key == "main_model":
                    dm_main = (np.nan, np.nan, np.nan, np.nan)
                else:
                    dm_main = (
                        *dm_test(m["mse_vec"], main_m["mse_vec"], h_cfg["dm_bandwidth"]),
                        *dm_test(m["qlike_vec"], main_m["qlike_vec"], h_cfg["dm_bandwidth"]),
                    )
                row = {
                    "asset": asset, "news_asset": news_asset, "window": window_num, "horizon": horizon, "model": name,
                    "MSE": m["MSE"], "QLIKE": m["QLIKE"], "R2": m["R2"], "DA": m["DA"],
                    "DM_MSE_t_vs_TS_MLP": dm_ts[0], "DM_MSE_p_vs_TS_MLP": dm_ts[1],
                    "DM_QLIKE_t_vs_TS_MLP": dm_ts[2], "DM_QLIKE_p_vs_TS_MLP": dm_ts[3],
                    "DM_MSE_t_vs_Main": dm_main[0], "DM_MSE_p_vs_Main": dm_main[1],
                    "DM_QLIKE_t_vs_Main": dm_main[2], "DM_QLIKE_p_vs_Main": dm_main[3],
                    "N": len(actual), "n_windows": 1,
                }
                rows.append(row)
                h_rows.append(row)
                pooled[key]["actual"].append(actual)
                pooled[key]["pred"].append(pred)

        if not h_rows:
            continue

        pooled_actual = np.concatenate(pooled["ts_mlp"]["actual"])
        ts_m = compute_metrics(np.concatenate(pooled["ts_mlp"]["pred"]), pooled_actual)
        main_m = compute_metrics(np.concatenate(pooled["main_model"]["pred"]), pooled_actual)
        h_df = pd.DataFrame(h_rows)
        for key, name in model_keys.items():
            pred = np.concatenate(pooled[key]["pred"])
            pooled_m = compute_metrics(pred, pooled_actual)
            model_rows = h_df[h_df["model"] == name]
            if key == "ts_mlp":
                dm_ts = (np.nan, np.nan, np.nan, np.nan)
            else:
                dm_ts = (
                    *dm_test(pooled_m["mse_vec"], ts_m["mse_vec"], h_cfg["dm_bandwidth"]),
                    *dm_test(pooled_m["qlike_vec"], ts_m["qlike_vec"], h_cfg["dm_bandwidth"]),
                )
            if key == "main_model":
                dm_main = (np.nan, np.nan, np.nan, np.nan)
            else:
                dm_main = (
                    *dm_test(pooled_m["mse_vec"], main_m["mse_vec"], h_cfg["dm_bandwidth"]),
                    *dm_test(pooled_m["qlike_vec"], main_m["qlike_vec"], h_cfg["dm_bandwidth"]),
                )
            rows.append({
                "asset": asset, "news_asset": news_asset, "window": "mean", "horizon": horizon, "model": name,
                "MSE": float(model_rows["MSE"].mean()),
                "QLIKE": float(model_rows["QLIKE"].mean()),
                "R2": float(model_rows["R2"].mean()),
                "DA": float(model_rows["DA"].mean()),
                "DM_MSE_t_vs_TS_MLP": dm_ts[0], "DM_MSE_p_vs_TS_MLP": dm_ts[1],
                "DM_QLIKE_t_vs_TS_MLP": dm_ts[2], "DM_QLIKE_p_vs_TS_MLP": dm_ts[3],
                "DM_MSE_t_vs_Main": dm_main[0], "DM_MSE_p_vs_Main": dm_main[1],
                "DM_QLIKE_t_vs_Main": dm_main[2], "DM_QLIKE_p_vs_Main": dm_main[3],
                "N": int(model_rows["N"].sum()), "n_windows": int(model_rows["window"].nunique()),
            })

    if not rows:
        raise FileNotFoundError(f"No prediction files found in {pred_dir}")

    out_path = os.path.join(results_dir, "dm_test_results.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, float_format="%.10f")
    log(f"Saved: {out_path}")
    cols = [
        "news_asset", "horizon", "model", "MSE", "QLIKE", "R2", "DA",
        "DM_MSE_t_vs_TS_MLP", "DM_MSE_p_vs_TS_MLP",
        "DM_QLIKE_t_vs_TS_MLP", "DM_QLIKE_p_vs_TS_MLP",
    ]
    log(df[df["window"].astype(str) == "mean"][cols].to_string(index=False))


def parse_main_args(default_asset, default_news_asset=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-dm", action="store_true")
    parser.add_argument("--rebuild-topics", action="store_true")
    parser.add_argument("--dm-only", action="store_true")
    parser.add_argument("--news-asset", default=default_news_asset)
    parser.add_argument("--horizons", nargs="+", choices=list(HORIZONS.keys()), default=None)
    parser.add_argument("--windows", nargs="+", type=int, default=None)
    args = parser.parse_args()
    if args.dm_only:
        run_dm_evaluation(default_asset, args.results_dir, news_asset=args.news_asset, horizons=args.horizons)
    else:
        run_expanding_training(
            default_asset, args.results_dir, args.batch_size, not args.skip_dm,
            args.rebuild_topics, news_asset=args.news_asset, horizons=args.horizons, windows=args.windows,
        )
