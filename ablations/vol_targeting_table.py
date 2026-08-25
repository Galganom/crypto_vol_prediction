"""Render LaTeX summary table for the vol-targeting backtest."""
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY = REPO_ROOT / "results/strategy/summary_btc_eth.csv"
OUT = REPO_ROOT / "data/results/vol_targeting_table.tex"
OUT_CSV = REPO_ROOT / "data/results/vol_targeting_metrics.csv"


def main():
    df = pd.read_csv(SUMMARY)
    df = df[["asset", "model", "sharpe", "ann_ret", "ann_vol",
             "max_dd", "vol_ratio", "turnover_avg"]]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")

    pivot_order = ["Oracle (actual RV)", "Buy & Hold", "HAR", "EGARCH",
                   "EGARCH+NED", "TS MLP", "Main Model"]
    btc = df[df["asset"] == "BTC"].set_index("model").reindex(pivot_order)
    eth = df[df["asset"] == "ETH"].set_index("model").reindex(pivot_order)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Vol-targeting backtest (4h rebalance, target 20\% ann.\ vol, 3\(\times\) leverage cap, 5\,bps cost per side).}",
        r"\label{tab:voltarget}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & Sharpe & Ann.\ ret. & Ann.\ vol. & Max DD & Turnover \\",
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{BTC --- 104{,}646 4h bars (W3--W8)}} \\",
    ]
    for m in pivot_order:
        r = btc.loc[m]
        lines.append(
            f"{m} & {r['sharpe']:.2f} & {r['ann_ret']*100:.1f}\\% & "
            f"{r['ann_vol']*100:.1f}\\% & {r['max_dd']*100:.0f}\\% & {r['turnover_avg']:.3f} \\\\"
        )
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\textit{ETH --- 104{,}641 4h bars (W3--W8)}} \\")
    for m in pivot_order:
        r = eth.loc[m]
        lines.append(
            f"{m} & {r['sharpe']:.2f} & {r['ann_ret']*100:.1f}\\% & "
            f"{r['ann_vol']*100:.1f}\\% & {r['max_dd']*100:.0f}\\% & {r['turnover_avg']:.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
