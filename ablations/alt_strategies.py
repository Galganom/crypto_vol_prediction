"""Alternative portfolios that USE the vol forecast as a signal, not a sizer.

For BTC + ETH, 4h horizon, no transaction costs (gross). Pooled test rows from
the 6 production windows.

Strategies (compared head-to-head against B&H):

  A. Vol-Stop B&H            — w=1 when pred_vol below 80th-pctile, else 0.
                                  ⇢ keeps B&H upside, clips tail by sitting out
                                    the top-quintile-vol bars (the crash bars).

  B. Vol-Regime Trend         — long when 72h-MA(log_ret) > 0, short when < 0,
                                  scaled to 0.5× when pred_vol > 80th-pctile.
                                  ⇢ classic time-series momentum with our vol
                                    forecast acting as a regime de-risker.

  C. News-Event De-Risk       — w=1 unless |log(main/ts)| > 90th-pctile in the
                                  trailing 30-day window, then w=0.
                                  ⇢ Main-vs-TS gap is the "news adds extra vol"
                                    signal; sit out when news is screaming.

  D. Inverse-Vol B&H          — w ∝ 1/pred_vol, scaled so avg-w = 1
                                  (NOT a fixed vol target → no leverage cap).
                                  ⇢ buy-and-hold weighted by certainty; weak
                                    forecasts get less capital.

  E. Cross-Sectional Vol-Rank — pure long-short BTC vs ETH: each bar long the
                                  asset with LOWER predicted vol, short the
                                  higher (dollar-neutral). Cross-asset only.

For each strategy we report Sharpe, ann ret, ann vol, max DD against B&H.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from main_model_common import build_price_bars, align_bars_with_news, load_news_features  # noqa: E402

HORIZON_HOURS = 4
PERIODS_PER_YEAR = 365 * 24 / HORIZON_HOURS  # 2190 for 4h
ASSETS = {
    "BTC": REPO_ROOT / "results/main/btc_cryptobert_v2",
    "ETH": REPO_ROOT / "results/main/eth_cryptobert_v2",
}
OUT_DIR = REPO_ROOT / "results/strategy/alt_strategies"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_pooled(asset):
    asset_dir = ASSETS[asset]
    files = sorted(asset_dir.glob("predictions/window_*_4h_predictions.npz"))
    main, ts, har, ega, ned, actual, idx, dt = [], [], [], [], [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        main.append(d["main_model"]); ts.append(d["ts_mlp"])
        har.append(d["har"]); ega.append(d["egarch"]); ned.append(d["egarch_ned"])
        actual.append(d["actual"]); idx.append(d["test_idx"]); dt.append(d["datetime"])
    out = {
        "main_model": np.concatenate(main),
        "ts_mlp": np.concatenate(ts),
        "har": np.concatenate(har),
        "egarch": np.concatenate(ega),
        "egarch_ned": np.concatenate(ned),
        "actual": np.concatenate(actual),
        "test_idx": np.concatenate(idx),
        "datetime": pd.to_datetime(np.concatenate(dt), utc=True),
    }
    return out


def forward_4h_log_ret(asset):
    bars = build_price_bars(asset)
    news = load_news_features(asset)
    bars, _, _ = align_bars_with_news(bars, news["timestamps"])
    log_ret = bars["log_ret"].values.astype(np.float64)
    fwd = pd.Series(log_ret).shift(-1).rolling(16, min_periods=16).sum().shift(-15).values
    return fwd, bars


def metrics(port_ret, name=""):
    r = port_ret[np.isfinite(port_ret)]
    mean = r.mean(); std = r.std(ddof=1)
    ann_ret = mean * PERIODS_PER_YEAR
    ann_vol = std * np.sqrt(PERIODS_PER_YEAR)
    sharpe = ann_ret / max(ann_vol, 1e-12)
    cum = np.cumsum(r)
    peak = np.maximum.accumulate(cum)
    max_dd = abs((cum - peak).min())
    return {"strategy": name, "sharpe": float(sharpe), "ann_ret": float(ann_ret),
            "ann_vol": float(ann_vol), "max_dd": float(max_dd), "n_obs": int(len(r))}


def vol_stop_bnh(pred_vol, fwd, q=0.80):
    thr = np.nanpercentile(pred_vol, q * 100)
    w = (pred_vol < thr).astype(float)
    return w * fwd, w


def vol_regime_trend(pred_vol, fwd, bars, idx, ma_hours=72, q=0.80):
    # 4h bars: ma_hours / 4 lookback for trend sign
    n_bars_ma = max(1, ma_hours // HORIZON_HOURS)
    bar_log_ret = bars["log_ret"].values.astype(np.float64)
    # Sum log-rets over the trailing ma_hours window
    # log_ret is 15-min bar level; for the 4h-aligned trend we sum 4× ma_hours/4 = 16*ma_hours/4 = 4*ma_hours
    # Actually log_ret is 15-min returns. ma_hours of trend = 4 * ma_hours/4 * 16 = 4*16=64 etc. Let's just sum 4*ma_hours 15-min bars.
    ma_15min = ma_hours * 4
    trend_sum = pd.Series(bar_log_ret).rolling(ma_15min, min_periods=ma_15min).sum().values
    trend_sign_full = np.sign(trend_sum)
    trend_sign = trend_sign_full[idx]
    thr = np.nanpercentile(pred_vol, q * 100)
    scale = np.where(pred_vol < thr, 1.0, 0.5)
    w = trend_sign * scale
    return w * fwd, w


def news_event_derisk(main_pred, ts_pred, fwd, lookback_bars=180, q=0.90):
    gap = np.log(np.clip(main_pred, 1e-15, None)) - np.log(np.clip(ts_pred, 1e-15, None))
    abs_gap = np.abs(gap)
    # Rolling 30-day-ish 90th percentile (180 4h bars = 30 days)
    s = pd.Series(abs_gap)
    rolling_thr = s.rolling(lookback_bars, min_periods=20).quantile(q).values
    w = np.where(abs_gap >= rolling_thr, 0.0, 1.0).astype(float)
    return w * fwd, w


def inverse_vol_bnh(pred_vol, fwd):
    inv = 1.0 / np.clip(pred_vol, 1e-12, None)
    w = inv / np.nanmean(inv)        # avg w = 1
    return w * fwd, w


def cross_section_pair(btc_pred_vol, eth_pred_vol, btc_fwd, eth_fwd, dt_btc, dt_eth):
    """Dollar-neutral: at each bar long lower-vol asset, short higher-vol asset.
    Aligns BTC and ETH by datetime intersection."""
    btc_s = pd.Series(btc_pred_vol, index=dt_btc, name="btc_vol")
    eth_s = pd.Series(eth_pred_vol, index=dt_eth, name="eth_vol")
    btc_r = pd.Series(btc_fwd, index=dt_btc, name="btc_fwd")
    eth_r = pd.Series(eth_fwd, index=dt_eth, name="eth_fwd")
    df = pd.concat([btc_s, eth_s, btc_r, eth_r], axis=1).dropna()
    # +0.5 long lower-vol, -0.5 short higher-vol
    btc_lower = df["btc_vol"] < df["eth_vol"]
    long_ret = np.where(btc_lower, df["btc_fwd"], df["eth_fwd"])
    short_ret = np.where(btc_lower, df["eth_fwd"], df["btc_fwd"])
    port_ret = 0.5 * long_ret - 0.5 * short_ret
    return np.asarray(port_ret)


def run_asset(asset):
    print(f"\n========== {asset} ==========")
    pred = load_pooled(asset)
    fwd_full, bars = forward_4h_log_ret(asset)
    fwd = fwd_full[pred["test_idx"]]
    # pred RV → annualized vol
    pred_vol = np.sqrt(np.clip(pred["main_model"], 1e-12, None) * PERIODS_PER_YEAR)

    rows = []
    # Baseline B&H
    rows.append(metrics(fwd, "Buy & Hold"))

    # A. Vol-Stop B&H @ 80th pct of predicted vol
    pr, _ = vol_stop_bnh(pred_vol, fwd, q=0.80)
    rows.append(metrics(pr, "Vol-Stop B&H (80th)"))
    pr, _ = vol_stop_bnh(pred_vol, fwd, q=0.90)
    rows.append(metrics(pr, "Vol-Stop B&H (90th)"))

    # B. Vol-Regime Trend (72h MA, scale 0.5 if pred_vol > 80th)
    pr, _ = vol_regime_trend(pred_vol, fwd, bars, pred["test_idx"],
                             ma_hours=72, q=0.80)
    rows.append(metrics(pr, "Vol-Regime Trend (72h)"))

    # C. News-Event De-Risk
    pr, _ = news_event_derisk(pred["main_model"], pred["ts_mlp"], fwd,
                              lookback_bars=180, q=0.90)
    rows.append(metrics(pr, "News-Event De-Risk (90th)"))

    # D. Inverse-Vol B&H
    pr, _ = inverse_vol_bnh(pred_vol, fwd)
    rows.append(metrics(pr, "Inverse-Vol B&H"))

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print(df.to_string(index=False))
    df.to_csv(OUT_DIR / f"single_asset_{asset.lower()}.csv",
              index=False, float_format="%.6f")
    return pred, fwd, pred_vol


def main():
    btc_pred, btc_fwd, btc_pv = run_asset("BTC")
    eth_pred, eth_fwd, eth_pv = run_asset("ETH")

    # E. Cross-sectional pair (dollar-neutral)
    print("\n========== Cross-Asset ==========")
    pr = cross_section_pair(btc_pv, eth_pv, btc_fwd, eth_fwd,
                            btc_pred["datetime"], eth_pred["datetime"])
    rows = [metrics(pr, "Cross-Section Vol-Rank (BTC/ETH)")]
    # Also: 50/50 B&H benchmark
    btc_s = pd.Series(btc_fwd, index=btc_pred["datetime"])
    eth_s = pd.Series(eth_fwd, index=eth_pred["datetime"])
    bnh50 = pd.concat([btc_s, eth_s], axis=1).dropna().mean(axis=1).values
    rows.append(metrics(bnh50, "50/50 BTC-ETH B&H"))
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print(df.to_string(index=False))
    df.to_csv(OUT_DIR / "cross_asset.csv", index=False, float_format="%.6f")

    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
