"""Haystack-retrieval test: does a retention-biased decay-gate init remove the 'interference wall'
without a curriculum?  Mirrors run_trial() in dualgoose/scripts/long_retrieval_probe.py (armed_gdn,
2 layers, 2 heads, d=32, vocab 64, pairs 4, dense 4, AdamW lr 1e-3 cosine, clip 1.0), scaled to CPU:
batch 128, 2000 steps. Only difference between arms: initial bias of the decay-gate slots.
"""
import argparse, json, sys, time
from pathlib import Path

import torch


ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True, help="path to a clone of github.com/JIBSIL/dualgoose @ 8ca624a")
ap.add_argument("--arm", choices=["baseline", "retain", "nodecay"], required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--seq-len", type=int, default=64)
ap.add_argument("--retain-bias", type=float, default=4.0)  # sigmoid(4) ~= 0.982
ap.add_argument("--out", required=True)
args = ap.parse_args()
sys.path.insert(0, args.repo)
from dualgoose.diffusion import apply_subs_sample, masked_nll_loss
from dualgoose.model import build_tiny_denoiser
from dualgoose.toy_data import make_long_retrieval_batch

torch.set_num_threads(1)
V, L, PAIRS, NQ = 64, args.seq_len, 4, 4
mask_id = V - 1
torch.manual_seed(args.seed)
model = build_tiny_denoiser(V, L, d_model=32, n_layers=2, n_heads=2,
                            model_type="deltanet" if args.arm == "nodecay" else "gated_deltanet",
                            short_conv=4, merge="static", direction="bidirectional")

gated = [m for m in model.modules() if getattr(m, "gated", False) and hasattr(m, "proj")]
assert len(gated) == (0 if args.arm == "nodecay" else 4), len(gated)
assert args.arm != "nodecay" or sum(getattr(m, "short_conv", 0) > 0 for m in model.modules()) == 4
if args.arm == "retain":
    with torch.no_grad():
        for m in gated:
            m.proj.bias[3 * m.d_key + 1:].fill_(args.retain_bias)
init_decay = [float(torch.sigmoid(m.proj.bias[3 * m.d_key + 1:]).mean()) for m in gated]

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-4)
gen = torch.Generator().manual_seed(args.seed)
log, t0 = [], time.time()
for step in range(args.steps):
    b = make_long_retrieval_batch(args.batch, L, V, mask_id, pairs=PAIRS, n_queries=NQ, generator=gen)
    loss = masked_nll_loss(model(b.corrupted, b.t), b.clean, b.corrupted, mask_id, t=b.t)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 100 == 0 or step == args.steps - 1:
        log.append((step, round(float(loss), 4), round(time.time() - t0, 1)))
        print(args.arm, args.seed, log[-1], flush=True)

# eval at max gap (table at the start), plus the paper's rebinding control
model.eval()
egen = torch.Generator().manual_seed(args.seed + 10_000)
vpos = torch.tensor([2 * p + 1 for p in range(PAIRS)])
correct = total = rebind = 0
with torch.no_grad():
    for _ in range(8):
        b = make_long_retrieval_batch(256, L, V, mask_id, pairs=PAIRS, n_queries=NQ, generator=egen)
        m = b.target_mask
        pred = apply_subs_sample(model(b.corrupted, b.t), b.corrupted, mask_id)
        correct += int(pred[m].eq(b.clean[m]).sum()); total += int(m.sum())
        rc = b.corrupted.clone(); rc[:, vpos] = b.corrupted[:, vpos.roll(-1)]
        rpred = apply_subs_sample(model(rc, b.t), rc, mask_id)
        rebind += int(rpred[m].eq(b.clean[m]).sum())
final_decay = [float(torch.sigmoid(m.proj.bias[3 * m.d_key + 1:]).mean()) for m in gated]
res = dict(arm=args.arm, seed=args.seed, acc=correct / total, rebind_acc=rebind / total,
           init_decay=init_decay, final_decay_bias_sigmoid=final_decay, steps=args.steps,
           batch=args.batch, seq_len=L, log=log, minutes=round((time.time() - t0) / 60, 1))
Path(args.out).write_text(json.dumps(res, indent=1))
print("DONE", json.dumps({k: res[k] for k in ("arm", "seed", "acc", "rebind_acc", "minutes")}), flush=True)
