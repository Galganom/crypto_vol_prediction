# Ablations

Current retained ablations:

```bash
python ablations/btc_all6_p2_feature_compare.py
python ablations/btc_all6_p2_feature_compare.py
```

Outputs are written under `results/ablations/`, separate from `results/main/`.

This runs BTC 4h windows `4 5 6` by default, with three seeds. For each
window and seed, one shared P1 TS branch is reused by every static and decay
news variant. The all-six feature comparison is also P2-only.
