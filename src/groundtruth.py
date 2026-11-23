"""
Stage 7 — two claims, tested.

CLAIM 1  a probabilistic spine is more accurate than a deterministic one
CLAIM 2  a deterministic spine can serve as ground truth

Claim 1 is measurable directly: score both against the real person_id and
compare. Claim 2 is the dangerous one, because it cannot be tested from inside
a system that assumes it. Here we can, since we hold real truth.

The test for claim 2: build pseudo-truth by closing on email-hash equality —
exactly what a team does when it says "our deterministic matches are the
labels" — then score every strategy against BOTH truths and compare.

If pseudo-truth were sound, the two rankings would agree. Where they disagree,
the disagreement is measurement error introduced by the label, and it is not
random: it always favours whichever method most resembles the rule that
produced the labels.
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


def c2(n):
    return n * (n - 1) // 2


def close(node_ids, edges):
    d = DSU()
    for n in node_ids:
        d.find(n)
    for e in edges:
        d.union(e[0], e[1])
    lab = {}
    for n in node_ids:
        lab[n] = d.find(n)
    return lab


def edges_from_key(records, field):
    b = defaultdict(list)
    for r in records:
        if r[field]:
            b[r[field]].append(int(r["record_id"]))
    out = []
    for ids in b.values():
        if len(ids) > 1:
            ids.sort()
            for i in range(1, len(ids)):
                out.append((ids[0], ids[i]))
    return out


def score_against(labels_pred, labels_truth, node_ids):
    """Pairwise precision and recall of one labelling against another."""
    pred = defaultdict(list)
    for n in node_ids:
        pred[labels_pred[n]].append(n)
    tp = n_pred = 0
    for c in pred.values():
        n_pred += c2(len(c))
        cnt = Counter(labels_truth[n] for n in c)
        tp += sum(c2(v) for v in cnt.values())
    n_true = sum(c2(v) for v in Counter(labels_truth.values()).values())
    p = tp / n_pred if n_pred else 1.0
    r = tp / n_true if n_true else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def main():
    clean = list(csv.DictReader(open(OUT / "records_clean.csv")))
    raw = list(csv.DictReader(open(OUT / "records_raw.csv")))
    node_ids = [int(r["record_id"]) for r in clean]
    truth = {int(a): int(b) for a, b in list(csv.reader(open(OUT / "truth.csv")))[1:]}
    TH = math.log2(99)

    rows = list(csv.DictReader(open(OUT / "edges.csv")))
    ev = [(int(r["a"]), int(r["b"])) for r in rows if float(r["bits"]) >= TH]

    det_email = edges_from_key(clean, "email_hash")
    det_union = det_email + edges_from_key(clean, "phone")

    strategies = [
        ("deterministic, email hash", det_email),
        ("deterministic, email OR phone", det_union),
        ("evidence budget", ev),
    ]
    labellings = {name: close(node_ids, e) for name, e in strategies}

    # ---------------- CLAIM 1 ----------------
    print("STAGE 7 — TESTING THE TWO CLAIMS\n")
    print("CLAIM 1: a probabilistic spine is more accurate than a deterministic one")
    print("  scored against the real person_id\n")
    print(f"  {'spine':<32}{'precision':>11}{'recall':>9}{'F1':>8}")
    print("  " + "-" * 60)
    for name, _ in strategies:
        p, r, f = score_against(labellings[name], truth, node_ids)
        print(f"  {name:<32}{p:>11.3f}{r:>9.3f}{f:>8.3f}")

    # ---------------- CLAIM 2 ----------------
    pseudo = labellings["deterministic, email hash"]
    p, r, f = score_against(pseudo, truth, node_ids)
    print("\n\nCLAIM 2: a deterministic spine can serve as ground truth")
    print("  first, how good are those labels? Scored against real truth:\n")
    print(f"    label precision {p:.3f}   label recall {r:.3f}   F1 {f:.3f}")
    print(f"    {1-p:.1%} of the pairs it calls 'same person' are not,")
    print(f"    and it misses {1-r:.1%} of the pairs that are.\n")

    print("  now score every spine against BOTH, and compare.\n")
    print(f"  {'spine':<32}{'vs real truth':>22}{'vs det. labels':>22}")
    print(f"  {'':<32}{'prec':>8}{'recall':>7}{'F1':>7}{'prec':>8}{'recall':>7}{'F1':>7}")
    print("  " + "-" * 76)
    out = []
    for name, _ in strategies:
        pr, rr, fr = score_against(labellings[name], truth, node_ids)
        pp, rp, fp = score_against(labellings[name], pseudo, node_ids)
        out.append((name, fr, fp))
        print(f"  {name:<32}{pr:>8.3f}{rr:>7.3f}{fr:>7.3f}"
              f"{pp:>8.3f}{rp:>7.3f}{fp:>7.3f}")

    best_real = max(out, key=lambda t: t[1])[0]
    best_pseudo = max(out, key=lambda t: t[2])[0]
    print(f"\n  best against real truth        : {best_real}")
    print(f"  best against deterministic labels: {best_pseudo}")
    if best_real != best_pseudo:
        print("\n  The labels pick a different winner than reality does.")
    print("  Note the deterministic spine scores 1.000 against its own labels.")
    print("  It is being graded by the rule that produced it.")

    # where the distortion lives
    ev_lab = labellings["evidence budget"]
    split_ok = merged_wrong = 0
    by_pseudo = defaultdict(list)
    for n in node_ids:
        by_pseudo[pseudo[n]].append(n)
    for cluster in by_pseudo.values():
        if len(cluster) < 2:
            continue
        real = {truth[n] for n in cluster}
        pred = {ev_lab[n] for n in cluster}
        if len(real) > 1 and len(pred) > 1:
            split_ok += 1          # label says one person, truth and budget disagree
        elif len(real) > 1 and len(pred) == 1:
            merged_wrong += 1
    print(f"\n  {split_ok:,} clusters where the labels say 'one person', the truth says")
    print("  otherwise, and the evidence budget correctly split them. Every one of")
    print("  those is scored as an ERROR when you grade against deterministic labels.")


if __name__ == "__main__":
    main()
