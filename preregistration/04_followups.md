# Round 3 predictions (written 2026-10-19 ~14:50 ET, before running)
Setup: same as cloud round 1 (L=64, 4 pairs, dense 4, armed_gdn 2 layers d=32, batch 128, 2000 steps, no curriculum).
Lock-in = acc >= 0.9. Rebind must collapse (~0.02) for locked runs.
Reference: round 1 bidirectional default 0/6, retain(+4) 6/6; decay-free 3/3.

Test 1 - standard-library init (r = exp(-A*dt), A~U(0,16), dt~logU[1e-3,1e-1]; median 0.94, 26% of channels >= 0.98),
bidirectional, plain haystack, 5 seeds.
  Prediction: locks in >= 4/5.
  If <= 2/5: the paper's wall would survive the standard init -> "needs more retention than today's defaults" headline.

Test 2 - one-direction (causal) cells, decay bias +4:
  2a plain haystack, 3 seeds (positive control): >= 2/3 lock in.
  2b collision decoys, 4 seeds: predicted to FAIL (<= 1/4), because a forward-only delta cell overwrites on key reuse
     and keeps the last (junk) binding.
  2c collision decoys with default init, 3 seeds: 0/3 (reference).
  If 2b locks in >= 3/4, the decoy problem is not open even for causal cells and the familiarity-gate idea has no target here.
