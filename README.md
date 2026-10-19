# The "interference wall" is a decay-gate initialization artifact

**Short version.** Boesch & Wee ([arXiv 2609.16183](https://arxiv.org/abs/2609.16183)) report that
fixed-state recurrent cells (gated delta rule, Mamba-2, RWKV-7) sit at chance when retrieving 4
key-value pairs across a distractor haystack, flat across lengths and architectures. They attribute
this to *interference under sparse supervision*, break it with a distance curriculum, and describe
training as a seed *lock-in lottery*.

In their public code ([JIBSIL/dualgoose @ 8ca624a](https://github.com/JIBSIL/dualgoose/tree/8ca624a465e22a26583716399ef9a216da1f5627)),
every recurrent cell's decay gate starts at about **0.5 retention per token**. The gated delta-rule
cell uses `sigmoid(Linear(x))` with default bias; the Mamba-2 reference uses `A_log = dt_bias = 0`.
Standard implementations (NVlabs Gated DeltaNet, flash-linear-attention, official Mamba-2) start
near **0.94**. At 0.5 per token, about 1e-15 of a stored binding survives a 50-token haystack.
The gradient needed to learn retention is therefore effectively zero at every tested length, which
reproduces the length-flat, architecture-independent chance floor the paper uses to argue *against*
a decay explanation.

**Changing only the initial decay-gate bias removes the wall with no curriculum.** This holds at
the paper's full budget, at L = 256, on the collision-decoy variant, for causal (one-direction)
cells, and with the *standard library* initialization rather than our hand-picked value. A
decay-free cell also learns the task.

## Results

Lock-in = final accuracy ≥ 0.9; chance = 1/54 ≈ 0.019. Every locked-in run also passed the paper's
rebinding control (table values deranged → accuracy collapses to ~0.02), so all successes are
genuine in-context retrieval. No run found a shortcut.

| Experiment | Setup | Paper default init | Our change | Fisher p |
|---|---|---|---|---|
| Main replication (suite A) | L=64, batch 256, 3000 steps, 10 seeds | 1/10 | bias +4: **10/10** | 1.2e-4 |
| Long haystack (suite C) | L=256, batch 128, 3000 steps, 5 seeds | 0/5 | bias +4: **5/5**; bias +6: **5/5** | 0.008 each |
| Collision decoys (suite B) | L=64, decoys reuse table keys | 1/5 | bias +4: **5/5** | 0.048 |
| Pilot | L=64, batch 128, 2000 steps | 0/6 | bias +4: **6/6** | 0.002 |
| Standard library init | as pilot; r = exp(−A·dt), A~U(0,16), dt~logU[1e-3,1e-1] | 0/6 (pilot) | **5/5** | 0.002 |
| Decay-free control | as pilot; delta rule + conv, no decay gate | 0/6 (pilot) | **3/3** | 0.012 |
| Causal cells, collision decoys | as pilot, forward-only | 0/3 | bias +4: **4/4** | 0.029 |
| Causal cells, plain haystack | as pilot, forward-only | — | bias +4: **3/3** | — |

For reference, the paper reports that its uniform curriculum reaches 0/9 at L=256 and its shaped
ramp 4/5 (§6).

Locked-in runs with the retentive init learn by step ~300–600. Baseline either never escapes or, in
rare seeds (2 of 29 across all baseline runs), escapes late (~step 1400).

Note: the standard-init and decay-free rows compare against the pilot's baseline (0/6), which used
the same reduced budget but was run in an earlier batch.

## Pre-registered predictions, including the ones that failed

Predictions were written before each batch of runs (`preregistration/`).

| Prediction | Outcome |
|---|---|
| Pilot: +4 ≥ 4/6, baseline mostly fails | **Pass** (6/6 vs 0/6) |
| Suite A: +4 ≥ 7/10, baseline ≤ 3/10, p < .05 | **Pass** (10/10 vs 1/10) |
| Suite B: +4 will **not** fix collision decoys (≤ baseline + 1) | **Fail**. +4 went 5/5 |
| Suite C: baseline 0/5, bias +6 ≥ 3/5, ordering +6 ≥ +4 ≥ baseline | Pass, but bias +4 also went 5/5, so the predicted +6 vs +4 gap did not appear |
| Decay-free cell locks in ≥ 2/3 | **Pass** (3/3) |
| Standard library init ≥ 4/5 | **Pass** (5/5) |
| Causal cells + +4 will **not** fix collision decoys (≤ 1/4) | **Fail**. Went 4/4 (default init: 0/3) |

The two failed predictions both narrowed the paper's "interference" account further than we expected.
We had predicted that decoys reusing table keys would cause genuine overwrite interference that a
retention init cannot fix. Instead, once the init lets gradient through, both bidirectional and
causal cells learn to keep the first binding. That a causal cell can do this with two layers is
consistent with the paper's own observation that collision retrieval needs a two-layer
"mark-and-route" circuit. We have not inspected the learned circuit.

The L=256 result also corrects our own arithmetic. Bias +4 leaves only ~1% of the signal over 246
tokens, yet it suffices. What matters is that the initial gradient is not astronomically small
(1e-2 versus 1e-16), not that most of the binding survives.

## What this does and does not show

**Shows:** On this task, in this codebase, the wall, its length-flatness, its cross-architecture
uniformity, the collision lottery, and the L=256 boundary all trace to the 0.5 initial retention.
The library-standard initialization does not have the problem.

**Does not show:**
- **That the task needs a learned decay at all.** The decay-free cell also works, and in our runs
  the decay-gate *bias* barely moved during training (median 0.982 → 0.982 with +4; 0.94 → 0.90–0.97
  with standard init). The most conservative reading: decay initialized at 0.5 makes the task
  unlearnable, and this task does not require forgetting. Whether the gate is used input-dependently
  was not measured.
- **Costs elsewhere.** We did not test MQAR at high load, state tracking, or length extrapolation with
  the retentive init. Prior work links stronger initial decay to better length generalization
  (Tong et al. 2026).
- **Anything at scale.** Tiny models (d=32, 2 layers), synthetic task. The companion DreamingGoose
  conversions initialize differently (distillation first) and are out of scope.
- **The paper's Mamba-2 and RWKV-7 arms.** These share the 0.5 init, but we only re-ran the gated
  delta-rule cell.

## Prior art (the mechanism is not new)

- Forget gate at 0.5 → gradient halves per step; positive bias fixes it: Gers et al. 2000;
  [Jozefowicz et al. 2015](https://proceedings.mlr.press/v37/jozefowicz15.pdf).
- Init timescale should match dependency length: [Tallec & Ollivier 2018](https://arxiv.org/abs/1804.11188)
  (chrono init); [Gu et al. 2020](https://arxiv.org/abs/1910.09890).
- Near-no-decay init helps Mamba learn copying/recall: [Trockman et al. 2024](https://arxiv.org/abs/2410.11135)
  (mimetic init).
- Strong initial decay traps Mamba-2 on a memorization plateau in key-value retrieval:
  [Tong et al. 2026](https://arxiv.org/abs/2609.16540).
- Recurrent recall failure as an optimization problem: [Okpekpe & Orvieto 2025](https://arxiv.org/abs/2508.19029).

Our contribution is narrow. It identifies that a specific published phenomenon, and the curriculum
proposed to fix it, rest on a non-standard initialization, and it shows that the standard
initialization removes it.

## Reproduce

Requires Python ≥ 3.10 and PyTorch (CPU is fine; each run is single-threaded).

```bash
git clone https://github.com/JIBSIL/dualgoose.git && git -C dualgoose checkout 8ca624a465e22a26583716399ef9a216da1f5627
# main suite (45 runs; resumable; written for macOS: uses caffeinate + sysctl)
bash code/run.sh
# single runs
python code/run_arm.py      --repo dualgoose --arm retain --retain-bias 4 --seed 0 --out r.json
python code/run_nodecay.py  --repo dualgoose --arm nodecay --seed 0 --out r.json
python code/run_followups.py --repo dualgoose --init std --seed 0 --out r.json
python code/run_followups.py --repo dualgoose --init retain --direction forward --collision --seed 0 --out r.json
```

Each JSON in `results/` records the arm, seed, final accuracy, rebinding-control accuracy, initial
(and, for follow-ups, final) decay-gate retention, and the training-loss log.
`results/full_suite/REPORT.txt` summarizes the 45-run suite; its per-run JSONs are in the same folder.

## Files

```
code/               experiment scripts (import the paper's cells unchanged; only the gate bias differs)
preregistration/    predictions, written before each batch of runs
results/            per-run JSONs: pilot/, full_suite/, decay_free/, followups/
```
