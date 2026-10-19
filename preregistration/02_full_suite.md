# Haystack decay-init test: pre-registered predictions

Written 2026-10-18, before any of these runs started.

## Claim under test
Boesch & Wee (arXiv 2609.16183) report that fixed-state recurrent cells sit at chance when
retrieving 4 key-value pairs across a distractor haystack, and attribute it to interference under
sparse supervision, fixed by a distance curriculum. Our alternative: it is mostly an
**initialization** artifact. Their cells' decay gate starts at sigmoid(~0) = 0.5 per token, so after
a 54-token haystack ~1e-16 of the table survives and the gradient needed to learn retention is ~0.
Starting the gate at "remember by default" (bias +4 -> 0.982) should remove the wall with no curriculum.

Only change between arms: the initial bias of the decay-gate slots. Everything else matches the
paper's haystack command (armed_gdn, 2 layers, 2 heads, d=32, 4 pairs, dense 4, AdamW 1e-3 cosine).
Lock-in = final accuracy >= 0.9 (chance 0.019). Rebinding control must collapse for locked runs.

## Predictions (pass / fail decided in advance)
**A. L=64, 10 seeds per arm (main test).**
Retain(bias 4) locks in on >= 7/10; baseline <= 3/10; Fisher p < 0.05. Locked runs' rebind accuracy < 0.1.
*Falsified if* retain(bias 4) is not significantly above baseline.

**B. L=64 collision decoys (decoys reuse table keys with wrong values), 5 seeds per arm.**
Retain(bias 4) does NOT meaningfully help: lock-in <= baseline + 1. Decoys overwrite rather than fade,
so a "keep more" init should not address them.
*If retain fixes collision too, our mechanism story is incomplete.*

**C. L=256, 5 seeds per arm (paper: uniform curriculum 0/9, shaped ramp 4/5).**
Baseline 0/5. Retain(bias 6, 0.9975/token, ~54% survives 246 tokens) locks in >= 3/5.
Ordering retain6 >= retain4 >= baseline (bias 4 leaves only ~1% after 246 tokens).
*If retain(bias 6) fails at L=256, init explains the short-range wall but long range still needs
something more (e.g. the paper's curriculum).*

## Notes
- Pilot (separate machine, before this suite): L=64, batch 128, 2000 steps; first retain seed locked in
  by step ~500 while baseline sat at chance through step 1200. One seed per arm, not evidence on its own.
- CPU on purpose: the paper's reference cell is a step-by-step loop over tiny matrices, where GPU
  kernel-launch overhead per timestep usually cancels any speedup (not measured here). Throughput comes from running one seed per core in parallel.
