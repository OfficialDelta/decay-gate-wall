# Pilot: criteria stated before running (2026-10-18, ~19:13 ET)

Stated in conversation before the first run started (not committed to a file at the time):

- Setup: armed_gdn (dualgoose @ 8ca624a), 2 layers, 2 heads, d=32, L=64, 4 pairs, dense 4,
  batch 128, 2000 steps, AdamW 1e-3 cosine, no curriculum. Arms: default init vs decay-gate bias +4.
- **Supports the hypothesis:** retain(+4) locks in (acc >= 0.9) on >= 4/6 seeds while baseline stays
  near the paper's ~1/10.
- **Falsifies it:** retain(+4) does no better than baseline.
- Rebinding control included so a "win" must be genuine key->value retrieval.

Result: baseline 0/6, retain(+4) 6/6 (Fisher two-sided p = 0.0022); rebind 0.017-0.020 for all 6
retain runs. Pass.
