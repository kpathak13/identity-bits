"""
Stage 6 — what does each added field actually buy?

The rest of the pipeline prices fields one at a time. This stage stacks them.

Fields are added in descending order of expected contribution. After each
addition the whole spine is rebuilt from scratch and scored against truth, so
every row is a real result rather than a projection.

Two things show up that a per-field table cannot show:

  1. The curve bends. The first few fields buy enormous amounts of recall. The
     last few buy almost nothing, because by then the budget is already met on
     most true pairs.

  2. A correlated field buys nothing at all. shared_id is
     derived from the email, so once the email is in the set they add reach
     without adding evidence. The greedy decorrelation in the scorer already
     charges them their overlap, and the spine barely moves.
"""
import csv
import math
from collections import defaultdict, Counter

from common import OUT, DSU, c2


def build_and_score(node_ids, pairs, by_id, w, over, prior, TH, fields, truth):
    """Score every candidate pair on `fields`, threshold, close, evaluate."""
    d = DSU()
    for n in node_ids:
        d.find(n)
    kept = 0
    for a, b in pairs:
        ra, rb = by_id[a], by_id[b]
        total = prior
        agreed = []
        for f in fields:
            va, vb = ra[f], rb[f]
            if not va or not vb:
                continue
            ag, dis = w[f]
            if va == vb:
                agreed.append(f)
            elif not math.isnan(dis):
                total += dis
        if agreed:
            agreed.sort(key=lambda f: -w[f][0])
            accepted = []
            for f in agreed:
                dup = max((over.get((f, g), over.get((g, f), 0.0))
                           for g in accepted), default=0.0)
                total += max(w[f][0] - dup, 0.0)
                accepted.append(f)
        if total >= TH:
            d.union(a, b)
            kept += 1

    comps = defaultdict(list)
    for n in node_ids:
        comps[d.find(n)].append(n)
    tp = pred = 0
    largest = 0
    for c in comps.values():
        largest = max(largest, len(c))
        pred += c2(len(c))
        cnt = Counter(truth[n] for n in c)
        tp += sum(c2(v) for v in cnt.values())
    actual = sum(c2(v) for v in Counter(truth.values()).values())
    p = tp / pred if pred else 1.0
    r = tp / actual
    return {"edges": kept, "clusters": len(comps), "largest": largest,
            "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


def main():
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="",
                    help="comma-separated subset to restrict the experiment to")
    ap.add_argument("--out", default="stacking.csv",
                    help="output file name under out/ (default stacking.csv)")
    args, _ = ap.parse_known_args()
    subset = [f.strip() for f in args.fields.split(",") if f.strip()]

    records = list(csv.DictReader(open(OUT / "records_clean.csv")))
    for r in records:
        r["record_id"] = int(r["record_id"])
    by_id = {r["record_id"]: r for r in records}
    node_ids = list(by_id)
    truth = {int(a): int(b) for a, b in list(csv.reader(open(OUT / "truth.csv")))[1:]}
    prior = float(open(OUT / "prior.txt").read())
    budget = -prior + math.log2(99)      # evidence required, for reporting
    TH = math.log2(99)                   # W already contains the prior

    wrows = [r for r in csv.DictReader(open(OUT / "weights.csv"))
             if r["field"] != "email"]          # plaintext duplicate of email_hash
    w = {r["field"]: (float(r["agree_bits"]), float(r["disagree_bits"]))
         for r in wrows}
    if subset:
        wrows = [r for r in wrows if r["field"] in subset]
    order = [r["field"] for r in
             sorted(wrows, key=lambda r: -float(r["expected_bits"]))]
    expected = {r["field"]: float(r["expected_bits"]) for r in wrows}

    over = {}
    for r in csv.DictReader(open(OUT / "dependence.csv")):
        v = float(r["overcount_bits"])
        if v > 0:
            over[(r["a"], r["b"])] = v

    pairs = [(int(r["a"]), int(r["b"])) for r in csv.DictReader(open(OUT / "edges.csv"))]

    print("STAGE 6 — WHAT EACH ADDED FIELD BUYS")
    print(f"  budget {budget:.1f} bits of evidence, i.e. W >= {TH:.1f}."
          f" Fields added by expected contribution.")
    print("  the spine is rebuilt and rescored from scratch at every step.\n")
    print(f"  {'+ field':<14}{'exp bits':>9}{'cum bits':>10}{'prec':>8}{'recall':>8}"
          f"{'F1':>7}{'ΔF1':>8}{'largest':>9}")
    print("  " + "-" * 73)

    rows, cum, prev_f1 = [], 0.0, 0.0
    for i, f in enumerate(order, 1):
        cum += expected[f]
        s = build_and_score(node_ids, pairs, by_id, w, over, prior, TH,
                            order[:i], truth)
        d = s["f1"] - prev_f1
        flag = ""
        if i > 1 and d < 0.005:
            flag = "   <-- buys nothing"
        print(f"  {f:<14}{expected[f]:>9.1f}{cum:>10.1f}{s['precision']:>8.3f}"
              f"{s['recall']:>8.3f}{s['f1']:>7.3f}{d:>+8.3f}{s['largest']:>9,}{flag}")
        rows.append({"n_fields": i, "field_added": f,
                     "expected_bits": expected[f], "cumulative_bits": cum, **s})
        prev_f1 = s["f1"]

    with open(OUT / args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ink, ind, fault, rule = "#15201D", "#5B5FA6", "#A8332B", "#C6D0CC"
    fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=150)
    x = [r["n_fields"] for r in rows]
    ax.plot(x, [r["recall"] for r in rows], color=ind, lw=2.5, marker="o", ms=4,
            label="recall")
    ax.plot(x, [r["precision"] for r in rows], color=ink, lw=2.5, marker="o", ms=4,
            label="precision")
    ax.set_xticks(x)
    ax.set_xticklabels([r["field_added"] for r in rows], rotation=45,
                       ha="right", fontsize=8)
    ax.set_ylabel("pairwise precision / recall")
    ax.set_ylim(0, 1.05)
    ax.set_title("What each added field buys the spine", loc="left", fontsize=13)
    ax.grid(color=rule, lw=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / args.out.replace(".csv", ".png"))
    print(f"\n  chart -> {OUT/args.out.replace('.csv','.png')}")


if __name__ == "__main__":
    main()
