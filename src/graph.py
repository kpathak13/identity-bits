"""
Stage 3 — candidate generation and scoring.

BLOCKING decides which pairs get looked at. It is not free: any true pair the
blocker drops is lost permanently, however good the scorer is. Blocking recall
is reported so that cost is visible.

Digital identifiers make good blocking keys precisely because they are sparse.
IP does not: households and carrier NAT put hundreds of records behind one
address, so IP blocks blow past MAX_BLOCK and get skipped.

SCORING assigns each surviving pair a total weight in bits:

    W = prior_bits + sum over populated fields of (agree or disagree weight)

Two scores are produced for every pair.

  naive      every field, weights summed as if independent
  corrected  a field set chosen so the pairwise overcount stays small

The difference between them is the cost of ignoring dependence. shared_id and
shared_id is derived from the email, and ip is a second reading of the same
household as postal, so a naive sum counts that evidence two or three times.
"""
import csv
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

MAX_BLOCK = 60          # a block bigger than this is a defect, not a signal

BLOCK_KEYS = [
    ("email",        lambda r: r["email"] or None),
    ("email_hash",   lambda r: r["email_hash"] or None),
    ("phone",        lambda r: r["phone"] or None),
    ("device_id",    lambda r: r["device_id"] or None),
    ("fpc",          lambda r: r["fpc"] or None),
    ("shared_id",    lambda r: r["shared_id"] or None),
    ("ip",           lambda r: r["ip"] or None),
    ("dob_postal",   lambda r: f"{r['dob']}|{r['postal']}" if r["dob"] and r["postal"] else None),
    ("sur_postal",   lambda r: f"{r['surname']}|{r['postal']}" if r["surname"] and r["postal"] else None),
    ("sur_giv_dob",  lambda r: f"{r['surname']}|{r['given_name']}|{r['dob']}"
                     if r["surname"] and r["given_name"] and r["dob"] else None),
]

# The corrected score keeps every field but subtracts the measured pairwise
# overcount whenever two correlated fields agree at the same time.


def candidates(records):
    pairs, stats = set(), []
    for name, keyfn in BLOCK_KEYS:
        buckets = defaultdict(list)
        for r in records:
            k = keyfn(r)
            if k:
                buckets[k].append(r["record_id"])
        added = skipped = biggest = 0
        for ids in buckets.values():
            biggest = max(biggest, len(ids))
            if len(ids) > MAX_BLOCK:
                skipped += 1
                continue
            for a, b in combinations(sorted(ids), 2):
                pairs.add((a, b))
                added += 1
        stats.append({"key": name, "blocks": len(buckets), "biggest_block": biggest,
                      "oversized_skipped": skipped, "pairs": added})
    return pairs, stats


def score(pairs, by_id, weights, prior, fields, overcount=None):
    """
    Sum the weights, then subtract the measured overcount for every pair of
    fields that BOTH agreed. That correction is what stops correlated evidence
    being counted twice, and it is measured from the data rather than guessed.
    """
    w = {r["field"]: (float(r["agree_bits"]), float(r["disagree_bits"]))
         for r in weights if r["field"] in fields}
    out = {}
    for a, b in pairs:
        ra, rb = by_id[a], by_id[b]
        total = prior
        agreed = []
        for f, (ag, dis) in w.items():
            va, vb = ra[f], rb[f]
            if not va or not vb:
                continue                    # missing contributes nothing
            if va == vb:
                total += ag
                agreed.append(f)
            elif not math.isnan(dis):
                total += dis
        if overcount and len(agreed) > 1:
            # Greedy decorrelation. Take the strongest agreeing field whole,
            # then charge each later field its largest overlap with a field
            # already counted. Subtracting every pair instead would subtract
            # k(k-1)/2 terms and bury true matches.
            agreed.sort(key=lambda f: -w[f][0])
            total -= sum(w[f][0] for f in agreed)          # undo the naive sum
            accepted = []
            for f in agreed:
                dup = max((overcount.get((f, g), overcount.get((g, f), 0.0))
                           for g in accepted), default=0.0)
                total += max(w[f][0] - dup, 0.0)
                accepted.append(f)
        out[(a, b)] = total
    return out


def main():
    records = list(csv.DictReader(open(OUT / "records_clean.csv")))
    for r in records:
        r["record_id"] = int(r["record_id"])
    by_id = {r["record_id"]: r for r in records}
    truth = {int(a): int(b) for a, b in list(csv.reader(open(OUT / "truth.csv")))[1:]}
    weights = list(csv.DictReader(open(OUT / "weights.csv")))
    prior = float(open(OUT / "prior.txt").read())
    all_fields = [r["field"] for r in weights]

    pairs, bstats = candidates(records)

    by_person = defaultdict(list)
    for r in records:
        by_person[truth[r["record_id"]]].append(r["record_id"])
    true_pairs = set()
    for ids in by_person.values():
        for a, b in combinations(sorted(ids), 2):
            true_pairs.add((a, b))
    recall = len(pairs & true_pairs) / len(true_pairs)
    n = len(records)
    all_possible = n * (n - 1) // 2

    over = {}
    for r in csv.DictReader(open(OUT / "dependence.csv")):
        v = float(r["overcount_bits"])
        if v > 0:
            over[(r["a"], r["b"])] = v

    naive = score(pairs, by_id, weights, prior, all_fields)
    corrected = score(pairs, by_id, weights, prior, all_fields, overcount=over)

    with open(OUT / "edges.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "bits", "bits_naive", "is_true"])
        for (a, b), c in corrected.items():
            w.writerow([a, b, f"{c:.4f}", f"{naive[(a,b)]:.4f}",
                        int(truth[a] == truth[b])])

    print("STAGE 3 — CANDIDATES AND SCORING")
    print(f"  {'blocking key':<14}{'blocks':>9}{'biggest':>9}{'skipped':>9}{'pairs':>11}")
    print("  " + "-" * 52)
    for s in bstats:
        note = "  <-- households and carrier NAT" if s["key"] == "ip" and s["oversized_skipped"] else ""
        print(f"  {s['key']:<14}{s['blocks']:>9,}{s['biggest_block']:>9,}"
              f"{s['oversized_skipped']:>9,}{s['pairs']:>11,}{note}")

    print(f"\n  candidate pairs {len(pairs):>13,}  of {all_possible:>16,} possible")
    print(f"  reduction       {all_possible/len(pairs):>13,.0f}x")
    print(f"  blocking recall {recall:>13.2%}   "
          f"({len(true_pairs)-len(pairs & true_pairs):,} true pairs lost, unrecoverable)")

    # how much does the naive sum inflate a pair?
    diffs = [naive[k] - corrected[k] for k in corrected]
    tp_diffs = [naive[k] - corrected[k] for k in corrected if truth[k[0]] != truth[k[1]]]
    print(f"\n  naive minus corrected, mean over all pairs      {sum(diffs)/len(diffs):>6.1f} bits")
    print(f"  naive minus corrected, mean over NON-matches    {sum(tp_diffs)/len(tp_diffs):>6.1f} bits")
    print("  inflating non-matches is how a naive sum manufactures false merges.")


if __name__ == "__main__":
    main()
