"""
Stage 5 — does the arithmetic actually predict reality?

Two checks.

CALIBRATION. For a pair with total weight W bits, the theory says

    predicted precision = 2^W / (1 + 2^W)

We bucket scored pairs by W and compare that prediction against the observed
fraction that really are the same person. If the two track, the budget is a
usable instrument rather than a story.

SWEEP. Move the threshold and watch precision, recall and the largest cluster
move together. This is where the cost of a low threshold becomes visible: the
largest component grows suddenly rather than gradually.
"""
import csv
import math
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def calibration(edges, n_buckets=12):
    """Observed vs predicted precision, bucketed by total weight."""
    lo = min(e[2] for e in edges)
    hi = max(e[2] for e in edges)
    width = (hi - lo) / n_buckets
    buckets = defaultdict(lambda: [0, 0])
    for a, b, w, is_true in edges:
        i = min(int((w - lo) / width), n_buckets - 1)
        buckets[i][0] += 1
        buckets[i][1] += is_true
    rows = []
    for i in sorted(buckets):
        n, t = buckets[i]
        centre = lo + width * (i + 0.5)
        pred = 2 ** centre / (1 + 2 ** centre) if centre < 200 else 1.0
        rows.append({"bits_mid": centre, "n": n,
                     "observed": t / n, "predicted": pred})
    return rows


def sweep(node_ids, edges, truth, thresholds):
    def c2(n):
        return n * (n - 1) // 2
    actual = sum(c2(v) for v in Counter(truth.values()).values())
    rows = []
    for th in thresholds:
        d = DSU()
        for n in node_ids:
            d.find(n)
        for a, b, w, _ in edges:
            if w >= th:
                d.union(a, b)
        comps = defaultdict(list)
        for n in node_ids:
            comps[d.find(n)].append(n)
        tp = pred = 0
        for c in comps.values():
            pred += c2(len(c))
            cnt = Counter(truth[n] for n in c)
            tp += sum(c2(v) for v in cnt.values())
        p = tp / pred if pred else 1.0
        r = tp / actual
        rows.append({"threshold": th, "precision": p, "recall": r,
                     "f1": 2 * p * r / (p + r) if p + r else 0.0,
                     "largest": max(len(c) for c in comps.values()),
                     "clusters": len(comps)})
    return rows


def charts(cal, sw, budget):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping charts)")
        return
    ink, ind, fault, rule = "#15201D", "#5B5FA6", "#A8332B", "#C6D0CC"

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    ax.plot([r["bits_mid"] for r in cal], [r["predicted"] for r in cal],
            color=ind, lw=2, ls="--", label="predicted  2^W / (1 + 2^W)")
    ax.plot([r["bits_mid"] for r in cal], [r["observed"] for r in cal],
            color=ink, lw=2.5, marker="o", ms=5, label="observed")
    ax.axvline(budget, color=fault, lw=1.5, ls=":")
    ax.text(budget + 0.4, 0.06, f"budget {budget:.1f}", color=fault, fontsize=9)
    ax.set_xlabel("total weight on the pair, bits")
    ax.set_ylabel("share of pairs that are the same person")
    ax.set_title("Does the arithmetic predict reality?", loc="left", fontsize=13)
    ax.grid(color=rule, lw=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    th = [r["threshold"] for r in sw]
    ax.plot(th, [r["precision"] for r in sw], color=ink, lw=2.5, label="precision")
    ax.plot(th, [r["recall"] for r in sw], color=ind, lw=2.5, label="recall")
    ax.axvline(budget, color=fault, lw=1.5, ls=":")
    ax.set_xlabel("threshold, bits")
    ax.set_ylabel("pairwise precision / recall")
    ax.set_ylim(0, 1.05)
    ax2 = ax.twinx()
    ax2.plot(th, [r["largest"] for r in sw], color=fault, lw=2, ls="--",
             label="largest cluster")
    ax2.set_ylabel("largest cluster, records", color=fault)
    ax2.set_yscale("log")
    ax.set_title("Lower the threshold and the biggest cluster runs away",
                 loc="left", fontsize=13)
    ax.grid(color=rule, lw=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="center left")
    for s in ("top",):
        ax.spines[s].set_visible(False)
        ax2.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "threshold_sweep.png")
    plt.close(fig)
    print(f"  charts -> {OUT/'calibration.png'}, {OUT/'threshold_sweep.png'}")


def main():
    edges = [(int(r["a"]), int(r["b"]), float(r["bits"]), int(r["is_true"]))
             for r in csv.DictReader(open(OUT / "edges.csv"))]
    node_ids = [int(r["record_id"]) for r in csv.DictReader(open(OUT / "records_clean.csv"))]
    truth = {int(r[0]): int(r[1]) for r in list(csv.reader(open(OUT / "truth.csv")))[1:]}
    prior = float(open(OUT / "prior.txt").read())
    budget = -prior + math.log2(99)

    print("STAGE 5 — EVALUATION\n")
    cal = calibration(edges)
    print("  calibration: predicted vs observed precision by weight")
    print(f"  {'bits':>8}{'pairs':>10}{'predicted':>12}{'observed':>11}{'gap':>8}")
    print("  " + "-" * 49)
    for r in cal:
        if r["n"] < 30:
            continue
        print(f"  {r['bits_mid']:>8.1f}{r['n']:>10,}{r['predicted']:>12.3f}"
              f"{r['observed']:>11.3f}{r['observed']-r['predicted']:>8.3f}")

    lo = math.floor(min(e[2] for e in edges))
    thresholds = [lo + i for i in range(0, int(max(e[2] for e in edges) - lo) + 2, 2)]
    sw = sweep(node_ids, edges, truth, thresholds)
    print(f"\n  threshold sweep (budget = {budget:.1f} bits)")
    print(f"  {'thresh':>8}{'precision':>11}{'recall':>9}{'F1':>8}"
          f"{'largest':>10}{'clusters':>11}")
    print("  " + "-" * 57)
    for r in sw:
        mark = "  <- budget" if abs(r["threshold"] - budget) < 1.0 else ""
        print(f"  {r['threshold']:>8.0f}{r['precision']:>11.3f}{r['recall']:>9.3f}"
              f"{r['f1']:>8.3f}{r['largest']:>10,}{r['clusters']:>11,}{mark}")

    best = max(sw, key=lambda r: r["f1"])
    at_budget = min(sw, key=lambda r: abs(r["threshold"] - budget))
    print(f"\n  the budget targets PRECISION, not F1.")
    print(f"    at the budget ({at_budget['threshold']:.0f} bits): "
          f"precision {at_budget['precision']:.3f}, recall {at_budget['recall']:.3f}, "
          f"F1 {at_budget['f1']:.3f}")
    print(f"    best F1 sits at {best['threshold']:.0f} bits: "
          f"precision {best['precision']:.3f}, recall {best['recall']:.3f}, "
          f"F1 {best['f1']:.3f}")
    print("    which threshold is right depends on what a false merge costs you.")

    with open(OUT / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sw[0].keys()))
        w.writeheader()
        w.writerows(sw)
    charts(cal, sw, budget)


if __name__ == "__main__":
    main()
