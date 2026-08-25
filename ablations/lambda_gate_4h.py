"""Learn a λ_max gate that zeros out fast-decaying articles, tuned on validation.

For each non-reactive article in a bar's 12-hour lookback, the model produces
(s_i, λ_i). The base event-signal is Σ_i s_i · exp(-λ_i · dt_i); main forecast
is ts_pred · exp(clip(W · signal + b, ±2)) where (W, b) are the residual head.

We learn a single scalar λ_max from validation and at inference set s_i = 0 for
any article with λ_i > λ_max — i.e. mute articles that decay faster than the
learned threshold. Tuned and evaluated chronologically per window.

Validation set: first 60% of each window's test bars (chronologically).
Held-out test: last 40%.

Reports MSE, QLIKE, log-MSE for: TS-only, Main (no gate), Main (gated), on the
held-out 40% of bars across all 6 windows.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from main_model_common import load_news_features
from main_model_cryptobert import CryptoBertMainModel

CB_PATH  = REPO_ROOT / "data/processed/btc_news_embeddings_cryptobert.npz"
RUN_CFG  = REPO_ROOT / "results/main/btc_cryptobert_v2/run_config.json"
CKPT_DIR = REPO_ROOT / "results/main/btc_cryptobert_v2/checkpoints"
PRED_DIR = REPO_ROOT / "results/main/btc_cryptobert_v2/predictions"
OUT_DIR  = REPO_ROOT / "results/analysis/btc_news_impact_v3"
LOG_EPS  = 1e-15
LOOKBACK_H = 12
VAL_FRACTION = 0.60   # first 60% of each window's bars = validation
HORIZON = "4h"


def metrics(pred, actual):
    p = np.clip(pred, LOG_EPS, None); a = np.clip(actual, LOG_EPS, None)
    mse     = float(((p - a) ** 2).mean())
    qlike   = float((a / p - np.log(a / p) - 1).mean())  # QLIKE = y/ŷ - log(y/ŷ) - 1
    log_mse = float(((np.log(p) - np.log(a)) ** 2).mean())
    return mse, qlike, log_mse


def build_records():
    cfg = json.loads(RUN_CFG.read_text())
    feat_dim = int(cfg.get("cryptobert_input_dim", 768))
    hidden = tuple(cfg["hidden"])

    cb = np.load(CB_PATH)
    si = np.argsort(cb["timestamps"])
    emb = cb["embeddings"][si].astype(np.float32)
    art_ts = cb["timestamps"][si].astype(np.int64)
    n = len(emb)
    news = load_news_features("BTC")
    nonreactive = news["nonreactive_mask"].astype(bool)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = []
    for w in range(1, 7):
        ckpt = CKPT_DIR / f"main_model_w{w}_{HORIZON}.pt"
        if not ckpt.exists(): continue
        m = CryptoBertMainModel(feat_dim, hidden=hidden).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        m.eval()
        s_all = np.zeros(n, dtype=np.float32)
        lam_all = np.zeros(n, dtype=np.float32)
        with torch.no_grad():
            B = 4096
            for i in range(0, n, B):
                x = torch.tensor(emb[i:i+B], dtype=torch.float32, device=device)
                out = m.news_head(m.input_norm(x))
                s_all[i:i+B] = torch.nn.functional.softplus(out[:, 0]).cpu().numpy()
                lam_all[i:i+B] = (0.001 + 0.499 * torch.sigmoid(out[:, 1])).cpu().numpy()
        W_res = float(m.residual_head.weight.detach().cpu().numpy().squeeze())
        b_res = float(m.residual_head.bias.detach().cpu().numpy().squeeze())

        pred = np.load(PRED_DIR / f"window_{w}_{HORIZON}_predictions.npz",
                       allow_pickle=True)
        pred_ts = (pd.to_datetime(pred["datetime"], utc=True).astype("int64") // 10**9).values
        ts_p = pred["ts_mlp"].astype(np.float64)
        mn_p = pred["main_model"].astype(np.float64)
        ac_p = pred["actual"].astype(np.float64)

        left = np.searchsorted(art_ts, pred_ts - LOOKBACK_H*3600, side="left")
        right = np.searchsorted(art_ts, pred_ts, side="left")
        for i in range(len(pred_ts)):
            l, r = left[i], right[i]
            arts = np.arange(l, r)
            arts = arts[nonreactive[arts]]
            records.append({
                "window": w, "bar_ts": int(pred_ts[i]),
                "ts_pred": float(ts_p[i]), "main_pred": float(mn_p[i]),
                "actual":  float(ac_p[i]),
                "W_res":   W_res, "b_res": b_res,
                "s":       s_all[arts].astype(np.float64),
                "lam":     lam_all[arts].astype(np.float64),
                "dt_min":  ((pred_ts[i] - art_ts[arts]).astype(np.float64) / 60.0),
            })
        del m
        if device.type == "cuda": torch.cuda.empty_cache()
        print(f"  W{w}: {len(pred_ts):,} test bars  |  residual_head W={W_res:.4f}, b={b_res:.4f}")
    return records


def gated_pred(rec, lambda_max):
    if len(rec["s"]) == 0:
        signal = 0.0
    else:
        keep = rec["lam"] <= lambda_max
        signal = float((rec["s"][keep] * np.exp(-rec["lam"][keep] * rec["dt_min"][keep])).sum())
    log_shift = np.clip(rec["W_res"] * signal + rec["b_res"], -2.0, 2.0)
    return rec["ts_pred"] * float(np.exp(log_shift))


def split_by_window(records, val_frac):
    """Per-window chronological split: first val_frac of each window's bars = val."""
    val, test = [], []
    by_w = {}
    for r in records:
        by_w.setdefault(r["window"], []).append(r)
    for w, rs in by_w.items():
        rs.sort(key=lambda r: r["bar_ts"])
        n = len(rs)
        cut = int(np.round(n * val_frac))
        val.extend(rs[:cut])
        test.extend(rs[cut:])
    return val, test


def main():
    print("Building per-bar lookback records ...")
    records = build_records()
    print(f"Total bars: {len(records):,}")
    val, test = split_by_window(records, VAL_FRACTION)
    print(f"Validation: {len(val):,}  |  Test: {len(test):,}\n")

    grid = np.unique(np.concatenate([
        np.array([1.0]),                                 # no gating (control)
        np.linspace(0.005, 0.05, 12),
        np.linspace(0.06, 0.20, 10),
        np.array([0.25, 0.30, 0.40, 0.50]),
    ]))
    grid.sort()

    val_actual = np.array([r["actual"] for r in val])
    print(f"{'λ_max':>8} {'val MSE×1e6':>14} {'val QLIKE':>11} {'val log-MSE':>12}  {'note':>6}")
    val_results = []
    for lm in grid:
        vp = np.array([gated_pred(r, lm) for r in val])
        mse, ql, lmse = metrics(vp, val_actual)
        val_results.append((lm, mse, ql, lmse))
        print(f"{lm:>8.3f} {mse*1e6:>14.4f} {ql:>11.5f} {lmse:>12.5f}"
              + ("  (no gate)" if lm >= 0.99 else ""))

    # Pick best (minimum val QLIKE)
    best = min(val_results, key=lambda t: t[2])
    best_lm = best[0]
    print(f"\nBest λ_max selected on val (min QLIKE): {best_lm:.3f}  "
          f"(val QLIKE={best[2]:.5f})")

    # Test metrics
    test_actual = np.array([r["actual"] for r in test])
    test_ts     = np.array([r["ts_pred"]   for r in test])
    test_main   = np.array([r["main_pred"] for r in test])
    test_gated  = np.array([gated_pred(r, best_lm) for r in test])

    mse_ts,    ql_ts,    lmse_ts    = metrics(test_ts,    test_actual)
    mse_main,  ql_main,  lmse_main  = metrics(test_main,  test_actual)
    mse_gated, ql_gated, lmse_gated = metrics(test_gated, test_actual)

    print("\n=== TEST RESULTS (BTC 4h, held-out last 40% of each window) ===")
    print(f"{'Model':<22s} {'MSE×1e6':>12s} {'QLIKE':>11s} {'log-MSE':>11s}")
    print(f"{'TS-only':<22s} {mse_ts*1e6:>12.4f} {ql_ts:>11.5f} {lmse_ts:>11.5f}")
    print(f"{'Main (no gate)':<22s} {mse_main*1e6:>12.4f} {ql_main:>11.5f} {lmse_main:>11.5f}")
    print(f"{'Main + λ-gate':<22s} {mse_gated*1e6:>12.4f} {ql_gated:>11.5f} {lmse_gated:>11.5f}")

    # Fraction of articles gated
    all_lams = np.concatenate([r["lam"] for r in records if len(r["lam"]) > 0])
    frac_gated = float((all_lams > best_lm).mean()) if len(all_lams) else 0.0
    med_hl_gated_min = float(np.median(np.log(2) / all_lams[all_lams > best_lm])) if (all_lams > best_lm).any() else float("nan")
    med_hl_kept_min  = float(np.median(np.log(2) / all_lams[all_lams <= best_lm])) if (all_lams <= best_lm).any() else float("nan")
    print(f"\nλ_max threshold: {best_lm:.3f}/min  →  half-life cutoff = {np.log(2)/best_lm:.1f} min")
    print(f"Fraction of non-reactive articles gated (s set to 0): {frac_gated*100:.1f}%")
    print(f"  median half-life of GATED articles: {med_hl_gated_min:.1f} min")
    print(f"  median half-life of KEPT  articles: {med_hl_kept_min:.1f} min")

    # Save summary
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"model": "TS-only", "MSE": mse_ts, "QLIKE": ql_ts, "logMSE": lmse_ts},
        {"model": "Main (no gate)", "MSE": mse_main, "QLIKE": ql_main, "logMSE": lmse_main},
        {"model": f"Main + λ-gate (λ_max={best_lm:.3f})", "MSE": mse_gated, "QLIKE": ql_gated, "logMSE": lmse_gated},
    ]).to_csv(OUT_DIR / "lambda_gate_4h_test_metrics.csv", index=False, float_format="%.8f")
    pd.DataFrame(val_results, columns=["lambda_max", "val_MSE", "val_QLIKE",
                                       "val_logMSE"]).to_csv(
        OUT_DIR / "lambda_gate_4h_val_grid.csv", index=False, float_format="%.10f")
    print(f"\nSaved: {OUT_DIR/'lambda_gate_4h_test_metrics.csv'}")
    print(f"Saved: {OUT_DIR/'lambda_gate_4h_val_grid.csv'}")


if __name__ == "__main__":
    main()
