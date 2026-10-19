"""Follow-up arms for the haystack decay-init test.

--init default : paper's init (decay gate = sigmoid(Linear(x)), bias ~0 -> ~0.5/token)
--init retain  : decay-gate bias filled with --retain-bias (4 -> 0.982)
--init std     : decay-gate bias set per channel so initial retention matches the standard
                 Gated DeltaNet / Mamba-2 library distribution, r = exp(-A*dt) with
                 A ~ U(0,16), dt ~ logU[1e-3, 1e-1] (median ~0.92), expressed in the paper's
                 own sigmoid parameterization (bias = logit(r)).
--direction    : bidirectional (paper's denoiser) or forward (causal only).
Everything else matches the earlier cloud runs (L=64, 4 pairs, dense 4, 2 layers, d=32, AdamW 1e-3
cosine, batch 128, 2000 steps).
"""
import argparse, json, math, sys, time
from pathlib import Path

import torch


ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True, help="path to a clone of github.com/JIBSIL/dualgoose @ 8ca624a")
ap.add_argument("--init", choices=["default", "retain", "std"], required=True)
ap.add_argument("--retain-bias", type=float, default=4.0)
ap.add_argument("--direction", choices=["bidirectional", "forward"], default="bidirectional")
ap.add_argument("--collision", action="store_true")
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--seq-len", type=int, default=64)
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
                            short_conv=4, merge="static", direction=args.direction)
gated = [m for m in model.modules() if getattr(m, "gated", False) and hasattr(m, "proj")]
assert len(gated) == 4, len(gated)  # forward mode builds but ignores the bwd cell

ginit = torch.Generator().manual_seed(args.seed + 777)  # separate stream: leaves model/data RNG untouched
with torch.no_grad():
    for m in gated:
        sl = m.proj.bias[3 * m.d_key + 1:]
        if args.init == "retain":
            sl.fill_(args.retain_bias)
        elif args.init == "std":
            n = sl.numel()
            A = torch.rand(n, generator=ginit) * 16.0
            dt = torch.exp(torch.rand(n, generator=ginit) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
            r = torch.exp(-A * dt).clamp(1e-4, 1 - 1e-6)
            sl.copy_(torch.log(r) - torch.log1p(-r))
    init_decay = [float(torch.sigmoid(m.proj.bias[3 * m.d_key + 1:]).median()) for m in gated]

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
        print(args.init, args.direction, args.seed, log[-1], flush=True)

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
with torch.no_grad():
    final_decay = [float(torch.sigmoid(m.proj.bias[3 * m.d_key + 1:]).median()) for m in gated]
res = dict(init=args.init, direction=args.direction, collision=args.collision, seed=args.seed,
           acc=correct / total, rebind_acc=rebind / total, init_decay_median=init_decay,
           final_decay_bias_median=final_decay, log=log, minutes=round((time.time() - t0) / 60, 1))
Path(args.out).write_text(json.dumps(res, indent=1))
print("DONE", json.dumps({k: res[k] for k in ("init", "direction", "collision", "seed", "acc",
                                              "rebind_acc", "minutes")}), flush=True)
