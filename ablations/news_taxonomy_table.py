"""Render a paper-ready LaTeX table for the BTC news taxonomy:
category, count, s_bar, half-life, impact share, two example headlines.

Reads:
  data/processed/btc_news_categories_clean.npz
  results/analysis/btc_news_impact_v3/category_s_lambda_stats.csv
  results/analysis/btc_news_impact_v3/category_impact_summary.csv

Writes:
  data/results/news_taxonomy_table.tex
  data/results/news_taxonomy_table.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical order (rank by signal magnitude → makes the table tell the story)
CAT_ORDER = [
    "Macro",
    "Geopolitical",
    "Newsletter_Series",
    "Regulatory_Legal",
    "Security_Crime",
    "OnChain_Flows",
    "Market_Commentary",
    "Other",
    "Opinion_Editorial",
    "Protocol_Infrastructure",
    "Institutional_Flows",
    "Product_Adoption",
]
DISPLAY = {c: c.replace("_", " ") for c in CAT_ORDER}

# Curated short headlines (hand-picked from the top-confidence examples to read
# well in a table; truncated to ~70 chars).
EXAMPLES = {
    "Macro": [
        "US Inflation Rises Faster Than Expected in March",
        "Fed Hikes Rates to Highest Since 2007; Bitcoin Slides Toward \\$19K",
    ],
    "Geopolitical": [
        "Bitcoin nears \\$63K as U.S. and Israel launch strikes on Iran",
        "Ukraine Update: Putin Recognizes Two Breakaway Regions",
    ],
    "Newsletter_Series": [
        "Crypto Long \\& Short: When ETF options start driving bitcoin",
        "MARKETS DAILY: ETF Arguments and China Token Troubles",
    ],
    "Regulatory_Legal": [
        "SEC Approves In-Kind Redemptions for All Spot Bitcoin and Ethereum ETFs",
        "Ripple's XRP Ruling a `Milestone Win' for Crypto Industry",
    ],
    "Security_Crime": [
        "Over \\$280M Drained in KuCoin Crypto Exchange Hack",
        "Crypto Exchange BigONE Confirms \\$27M Hack",
    ],
    "OnChain_Flows": [
        "Bitcoin Whales Wake Up From 14-Year Slumber to Move Over \\$2B of BTC",
        "FTX Contagion Grips Markets as Exchanges Scramble for Proof of Reserves",
    ],
    "Market_Commentary": [
        "Bitcoin Slides Below \\$38K, Support Is Above \\$35K",
        "This Metric Suggests Bitcoin Has Bottomed Out",
    ],
    "Other": [
        "Eminem Purchases Bored Ape Yacht Club NFT for \\$462K",
        "Elon Musk Says SpaceX Holds Bitcoin at `B Word' Conference",
    ],
    "Opinion_Editorial": [
        "Three Thoughts on the `Crypto Bubble'",
        "Crypto Has Parallels With Subprime Mortgage Crisis, Says Paul Krugman",
    ],
    "Protocol_Infrastructure": [
        "Bitcoin Halving Arrives: Mining Rewards Drop for Third Time in History",
        "Taproot, Bitcoin's Long-Anticipated Upgrade, Has Activated",
    ],
    "Institutional_Flows": [
        "MicroStrategy Splurges Another \\$650M on Latest Bitcoin Investment",
        "BlackRock's Bitcoin ETF Snaps 71-Day Inflow Streak",
    ],
    "Product_Adoption": [
        "Strike Launches Dollar-Cost Averaging Product in US",
        "Paxful Launches Tool Allowing Businesses to Receive Payment in Bitcoin",
    ],
}


def main():
    lab = np.load(REPO_ROOT / "data/processed/btc_news_categories_clean.npz",
                  allow_pickle=True)
    cats = np.array(lab["category"], dtype=object)
    counts = pd.Series(cats).value_counts().to_dict()

    sl_stats = pd.read_csv(
        REPO_ROOT / "results/analysis/btc_news_impact_v3/category_s_lambda_stats.csv"
    ).set_index("category")
    imp_stats = pd.read_csv(
        REPO_ROOT / "results/analysis/btc_news_impact_v3/category_impact_summary.csv"
    ).set_index("category")

    rows = []
    for c in CAT_ORDER:
        n = counts.get(c, 0)
        s_bar = sl_stats.loc[c, "s_weighted_mean"] if c in sl_stats.index else np.nan
        hl = sl_stats.loc[c, "half_life_weighted_median_min"] if c in sl_stats.index else np.nan
        imp = imp_stats.loc[c, "impact_share_abs"] if c in imp_stats.index else 0.0
        rows.append({"category": DISPLAY[c], "n": n, "s_bar": s_bar,
                     "half_life_min": hl, "impact_share_pct": imp * 100,
                     "example1": EXAMPLES[c][0], "example2": EXAMPLES[c][1]})
    df = pd.DataFrame(rows)
    out_csv = REPO_ROOT / "data/results/news_taxonomy_table.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, float_format="%.3f")

    # LaTeX: two-row-per-category compact form
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{BTC news taxonomy: per-category counts, model-implied signal "
        r"magnitude $\bar{s}$, median news half-life, share of total absolute "
        r"price-impact attributed by the signal-share decomposition over the 6 "
        r"production windows, and two example headlines.}",
        r"\label{tab:news-taxonomy}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrrp{0.50\linewidth}@{}}",
        r"\toprule",
        r"Category & $N$ & $\bar{s}$ & $t_{1/2}$ (min) & Impact \% & Example headlines \\",
        r"\midrule",
    ]
    for r in rows:
        n_str = f"{r['n']:,}"
        s_str = f"{r['s_bar']:.2f}" if pd.notna(r['s_bar']) else "--"
        hl_str = f"{r['half_life_min']:.0f}" if pd.notna(r['half_life_min']) else "--"
        imp_str = f"{r['impact_share_pct']:.1f}"
        ex = f"\\textit{{{r['example1']}}}; \\textit{{{r['example2']}}}"
        lines.append(
            f"{r['category']} & {n_str} & {s_str} & {hl_str} & {imp_str} & {ex} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_tex = REPO_ROOT / "data/results/news_taxonomy_table.tex"
    out_tex.write_text("\n".join(lines), encoding="utf-8")

    print(df.to_string(index=False))
    print(f"\nSaved CSV: {out_csv}")
    print(f"Saved LaTeX: {out_tex}")


if __name__ == "__main__":
    main()
