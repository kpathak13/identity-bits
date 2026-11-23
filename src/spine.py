"""
Stage 4 — from edges to a spine.

Transitive closure turns edges into resolved people, and that is where a single
bad edge stops being one bad merge and starts welding whole clusters together.

Strategies compared:

  A  deterministic, raw email        no hygiene at all
  B  deterministic, clean email      hygiene, still a rule
  C  clean email OR phone            a second key for recall
  D  device_id                       the digital equivalent of B
  E  fpc (first-party cookie)        high evidence, almost never agrees
  F  ip                              household-level, treated as person-level
  G  email OR shared_id              two names for one piece of evidence
  H  evidence budget, naive sum      every field, dependence ignored
  I  evidence budget, corrected      dependent fields removed

G is the one worth staring at. shared_id is derived from the
email, so the union adds reach without adding evidence, and every defect in the
email propagates into all three.

F is the other one. IP carries real bits, but they are bits about a household.

Pruning is then applied to every strategy. It only ever REMOVES edges.
"""
import csv
import math
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
MAX_PLAUSIBLE = 12          # more records than this for one person is suspect


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


def components(node_ids, edges):
    d = DSU()
    for n in node_ids:
        d.find(n)
    for e in edges:
        d.union(e[0], e[1])
    out = defaultdict(list)
    for n in node_ids:
        out[d.find(n)].append(n)
    return list(out.values())


def edges_from_key(records, keyfn, weight=999.0):
    buckets = defaultdict(list)
    for r in records:
        k = keyfn(r)
        if k:
            buckets[k].append(int(r["record_id"]))
    edges = []
    for ids in buckets.values():
        if len(ids) < 2:
            continue
        ids.sort()
        for i in range(1, len(ids)):
            edges.append((ids[0], ids[i], weight))
    return edges


def prune(node_ids, edges, max_size=MAX_PLAUSIBLE):
    d = DSU()
    for n in node_ids:
        d.find(n)
    size = Counter({n: 1 for n in node_ids})
    kept, dropped = [], []
    for a, b, w in sorted(edges, key=lambda e: -e[2]):
        ra, rb = d.find(a), d.find(b)
        if ra == rb:
            kept.append((a, b, w))
            continue
        if size[ra] + size[rb] > max_size:
            dropped.append((a, b, w))
            continue
        d.union(a, b)
        size[d.find(a)] = size[ra] + size[rb]
        kept.append((a, b, w))
    return kept, dropped


def c2(n):
    return n * (n - 1) // 2


def stats(comps, truth):
    sizes = [len(c) for c in comps]
    tp = pred = 0
    for c in comps:
        pred += c2(len(c))
        cnt = Counter(truth[n] for n in c)
        tp += sum(c2(v) for v in cnt.values())
    actual = sum(c2(v) for v in Counter(truth.values()).values())
    p = tp / pred if pred else 0.0
    r = tp / actual if actual else 0.0
    return {"clusters": len(comps), "largest": max(sizes),
            "over": sum(1 for s in sizes if s > MAX_PLAUSIBLE),
            "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


def main():
    raw = list(csv.DictReader(open(OUT / "records_raw.csv")))
    clean = list(csv.DictReader(open(OUT / "records_clean.csv")))
    node_ids = [int(r["record_id"]) for r in clean]
    truth = {int(a): int(b) for a, b in list(csv.reader(open(OUT / "truth.csv")))[1:]}
    prior = float(open(OUT / "prior.txt").read())
    budget = -prior + math.log2(99)      # evidence required, for reporting
    TH = math.log2(99)                   # W already contains the prior

    rows = list(csv.DictReader(open(OUT / "edges.csv")))
    ev_corrected = [(int(r["a"]), int(r["b"]), float(r["bits"]))
                    for r in rows if float(r["bits"]) >= TH]
    ev_naive = [(int(r["a"]), int(r["b"]), float(r["bits_naive"]))
                for r in rows if float(r["bits_naive"]) >= TH]

    key = lambda f: (lambda r: r[f] or None)
    strategies = [
        ("A  deterministic, raw email",
         edges_from_key(raw, lambda r: (r["email"] or "").strip().lower() or None)),
        ("B  deterministic, clean email", edges_from_key(clean, key("email"))),
        ("B2 deterministic, hashed email", edges_from_key(clean, key("email_hash"))),
        ("B3 hashed email, no hygiene at all",
         edges_from_key(raw, lambda r: r["email_sha256_canon"] or None)),
        ("C  clean email OR phone",
         edges_from_key(clean, key("email")) + edges_from_key(clean, key("phone"))),
        ("D  device_id", edges_from_key(clean, key("device_id"))),
        ("E  fpc (first-party cookie)", edges_from_key(clean, key("fpc"))),
        ("F  ip (household address)", edges_from_key(clean, key("ip"))),
        ("G  email OR shared_id",
         edges_from_key(clean, key("email")) + edges_from_key(clean, key("shared_id"))),
        (f"H  budget {budget:.0f} bits, naive sum", ev_naive),
        (f"I  budget {budget:.0f} bits, corrected", ev_corrected),
    ]

    hdr = (f"  {'strategy':<36}{'clusters':>10}{'largest':>9}{'>12':>6}"
           f"{'prec':>8}{'recall':>8}{'F1':>7}")

    print("STAGE 4 — SPINE")
    print(f"  budget {budget:.1f} bits of evidence (W >= {TH:.1f}) — corrected clears {len(ev_corrected):,} pairs,"
          f" naive clears {len(ev_naive):,}\n")
    print("  raw transitive closure")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    keep = {}
    for name, edges in strategies:
        s = stats(components(node_ids, edges), truth)
        keep[name] = edges
        print(f"  {name:<36}{s['clusters']:>10,}{s['largest']:>9,}{s['over']:>6,}"
              f"{s['precision']:>8.3f}{s['recall']:>8.3f}{s['f1']:>7.3f}")

    print(f"\n  after graph hygiene — prune components over {MAX_PLAUSIBLE} records,"
          f" weakest edges first")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, edges in strategies:
        kept, _ = prune(node_ids, edges)
        comps = components(node_ids, kept)
        s = stats(comps, truth)
        print(f"  {name:<36}{s['clusters']:>10,}{s['largest']:>9,}{s['over']:>6,}"
              f"{s['precision']:>8.3f}{s['recall']:>8.3f}{s['f1']:>7.3f}")
        if name.startswith("I"):
            with open(OUT / "spine.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["record_id", "spine_id"])
                for i, c in enumerate(comps):
                    for n in sorted(c):
                        w.writerow([n, i])
    print("\n  wrote out/spine.csv from strategy I after pruning")


if __name__ == "__main__":
    main()
