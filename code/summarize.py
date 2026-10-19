"""Summarize results/*.json into lock-in rates per condition, with a Fisher exact test vs baseline."""
import json, sys
from collections import defaultdict
from math import comb
from pathlib import Path

LOCK = 0.9
root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "results")
rows = [json.loads(p.read_text()) for p in sorted(root.glob("*.json"))]


def fisher_two_sided(a, b, c, d):
    """2x2 table [[a,b],[c,d]] -> two-sided Fisher exact p."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    def p(x):
        return comb(c1, x) * comb(n - c1, r1 - x) / comb(n, r1)
    p0 = p(a)
    lo, hi = max(0, r1 + c1 - n), min(r1, c1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 * (1 + 1e-9)))


groups = defaultdict(list)
for r in rows:
    task = f"L{r['seq_len']}{' collision' if r['collision'] else ''}"
    arm = r["arm"] if r["arm"] == "baseline" else f"retain(bias {r['retain_bias']:g})"
    groups[(task, arm)].append(r)

print(f"{'task':<16}{'arm':<20}{'n':>3}{'lock-in':>10}{'mean acc':>10}{'rebind':>9}   per-seed acc")
for task in sorted({t for t, _ in groups}):
    base = groups.get((task, "baseline"), [])
    bl = sum(r["acc"] >= LOCK for r in base)
    for (t, arm), rs in sorted(groups.items()):
        if t != task:
            continue
        rs = sorted(rs, key=lambda r: r["seed"])
        k = sum(r["acc"] >= LOCK for r in rs)
        accs = " ".join(f"{r['acc']:.2f}" for r in rs)
        line = (f"{task:<16}{arm:<20}{len(rs):>3}{f'{k}/{len(rs)}':>10}"
                f"{sum(r['acc'] for r in rs) / len(rs):>10.3f}"
                f"{sum(r['rebind_acc'] for r in rs) / len(rs):>9.3f}   {accs}")
        if arm != "baseline" and base:
            line += f"   (Fisher vs baseline p={fisher_two_sided(k, len(rs) - k, bl, len(base) - bl):.3g})"
        print(line)
print(f"\nlock-in = final accuracy >= {LOCK}. chance = 1/54 = 0.019. "
      "rebind should collapse toward ~0.03 for a genuine retriever.")
