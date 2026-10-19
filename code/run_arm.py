"""Haystack-retrieval test: does a retention-biased decay-gate init remove the 'interference wall'
of Boesch & Wee (arXiv 2609.16183) without a curriculum?

Mirrors run_trial() in dualgoose/scripts/long_retrieval_probe.py: armed_gdn (gated delta rule +
short conv, bidirectional), 2 layers, 2 heads, d=32, vocab 64, 4 pairs, dense supervision 4,
AdamW lr 1e-3 with cosine decay, grad clip 1.0, no curriculum. The ONLY difference between arms is
the initial bias of the decay-gate slots (baseline = PyTorch default ~0 -> decay ~0.5).
"""
import argparse, json, sys, time
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--arm", choices=["baseline", "retain"], required=True)
ap.add_argument("--retain-bias", type=float, default=4.0)  # sigmoid(4)=0.982, sigmoid(6)=0.9975
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--seq-len", type=int, default=64)
ap.add_argument("--collision", action="store_true")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--out", required=True)
args = ap.parse_args()

sys.path.insert(0, args.repo)
from dualgoose.diffusion import apply_subs_sample, masked_nll_loss  # noqa: E402
from dualgoose.model import build_tiny_denoiser  # noqa: E402
from dualgoose.toy_data import make_long_retrieval_batch  # noqa: E402

torch.set_num_threads(1)
V, L, PAIRS, NQ = 64, args.seq_len, 4, 4
mask_id = V - 1
torch.manual_seed(args.seed)
model = build_tiny_denoiser(V, L, d_model=32, n_layers=2, n_heads=2, model_type="gated_deltanet",
                            short_conv=4, merge="static", direction="bidirectional")
gated = [m for m in model.modules() if getattr(m, "gated", False) and hasattr(m, "proj")]
assert len(gated) == 4, len(gated)
if args.arm == "retain":
    with torch.no_grad():
        for m in gated:
            m.proj.bias[3 * m.d_key + 1:].fill_(args.retain_bias)
with torch.no_grad():
    init_decay = [float(torch.sigmoid(m.proj.bias[3 * m.d_key + 1:]).mean()) for m in gated]

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-4)
gen = torch.Generator().manual_seed(args.seed)
log, t0 = [], time.time()
for step in range(args.steps):
    b = make_long_retrieval_batch(args.batch, L, V, mask_id, pairs=PAIRS, n_queries=NQ,
                                  collision=args.collision, generator=gen)
    loss = masked_nll_loss(model(b.corrupted, b.t), b.clean, b.corrupted, mask_id, t=b.t)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 100 == 0 or step == args.steps - 1:
        log.append((step, round(float(loss), 4), round(time.time() - t0, 1)))
        print("STEP", step, log[-1][1], log[-1][2], flush=True)

# eval at max gap (table at the start) + the paper's rebinding control (deranged table values,
# scored against the ORIGINAL answer: a real retriever collapses toward the floor)
model.eval()
egen = torch.Generator().manual_seed(args.seed + 10_000)
vpos = torch.tensor([2 * p + 1 for p in range(PAIRS)])
correct = total = rebind = 0
with torch.no_grad():
    for _ in range(8):
        b = make_long_retrieval_batch(256, L, V, mask_id, pairs=PAIRS, n_queries=NQ,
                                      collision=args.collision, generator=egen)
        m = b.target_mask
        pred = apply_subs_sample(model(b.corrupted, b.t), b.corrupted, mask_id)
        correct += int(pred[m].eq(b.clean[m]).sum()); total += int(m.sum())
        rc = b.corrupted.clone(); rc[:, vpos] = b.corrupted[:, vpos.roll(-1)]
        rpred = apply_subs_sample(model(rc, b.t), rc, mask_id)
        rebind += int(rpred[m].eq(b.clean[m]).sum())
res = dict(arm=args.arm, retain_bias=args.retain_bias if args.arm == "retain" else None,
           seed=args.seed, seq_len=L, collision=args.collision, acc=correct / total,
           rebind_acc=rebind / total, init_decay=init_decay, steps=args.steps, batch=args.batch,
           log=log, minutes=round((time.time() - t0) / 60, 1))
Path(args.out).write_text(json.dumps(res, indent=1))
print("DONE", json.dumps({k: res[k] for k in ("arm", "seed", "seq_len", "collision", "acc",
                                              "rebind_acc", "minutes")}), flush=True)
