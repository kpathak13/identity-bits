"""
Stage 8 — the three claims the post makes in prose, computed.

Everything else in this repo measures one population. Three of the post's
statements are not about this population at all, and until now none of them
could be run:

  1. The worked example. 1B records over 250M people gives a -28.3 bit prior
     and a ~35 bit budget. This repo's corpus is smaller, so its budget is
     ~22. Both should fall out of the same function.

  2. u as collision entropy from a frequency table. weights.py measures u by
     counting cross-person pairs, which is correct but is not the derivation
     the post leads with. Here we price a field straight from the bundled
     Census and SSA tables using u = sum(p_i^2), and check the two agree.

  3. "Same hash. Different population. Different evidence." The punchline.
     The same identifier, hashed identically, is priced against two synthetic
     populations that differ only in how often one address covers more than
     one person.
"""
import csv
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out"


# ---------------------------------------------------------------- 1
def size_budget(n_records, n_people, target_precision=0.99):
    """Prior odds and the evidence budget for any corpus. Pure arithmetic."""
    r = n_records / n_people
    true_pairs = n_people * (r * (r - 1) / 2)
    all_pairs = n_records * (n_records - 1) / 2
    prior = true_pairs / (all_pairs - true_pairs)
    prior_bits = math.log2(prior)
    lift = math.log2(target_precision / (1 - target_precision))
    return prior_bits, lift, -prior_bits + lift


# ---------------------------------------------------------------- 2
def surname_ceiling_bounds(path):
    """
    Bounds on sum(p^2) for the whole population from a top-N table of raw
    population shares. Returns (lower, upper, coverage, largest_tail_share).
    """
    rows = list(csv.DictReader(open(path)))
    p = [float(r["freq"]) for r in rows]
    cover = sum(p)
    lo = sum(x * x for x in p)
    tail_max = min(p)                       # no tail name is more common than the N-th
    hi = lo + (1.0 - cover) * tail_max      # tail sum(p^2) <= tail mass x largest tail share
    return lo, hi, cover, tail_max


def u_from_frequency_table(path, name_col, freq_col):
    """u = sum(p_i^2), computed from a published frequency distribution."""
    rows = list(csv.DictReader(open(path)))
    f = [float(r[freq_col]) for r in rows]
    t = sum(f)
    p = [x / t for x in f]
    return sum(x * x for x in p)


# ---------------------------------------------------------------- 3
def price_key(n_people, share_rate, records_per_key=2, m=0.95):
    """
    Bits carried by an exact key match, as a function of how often one key
    covers more than one person. Nothing else differs between the runs.
    """
    pairs = n_people * (n_people - 1) / 2
    agreeing_cross = (share_rate * n_people / 2) * (records_per_key * (records_per_key - 1) / 2)
    u = max(agreeing_cross / pairs, 1e-15)
    return u, math.log2(m / u)


def main():
    print("STAGE 8 — THE POST'S PROSE CLAIMS, COMPUTED\n")

    print("1. THE WORKED EXAMPLE")
    print("   the same function sizes any corpus\n")
    print(f"   {'records':>12}{'people':>12}{'prior bits':>13}{'+99%':>8}{'budget':>9}")
    print("   " + "-" * 54)
    for nr, np_ in [(1e9, 2.5e8), (2.4e5, 1e5), (4.8e4, 2e4), (1e7, 3e6)]:
        pb, lift, bud = size_budget(nr, np_)
        tag = "  <-- the post" if nr == 1e9 else ""
        print(f"   {nr:>12,.0f}{np_:>12,.0f}{pb:>13.1f}{lift:>8.1f}{bud:>9.1f}{tag}")
    print("\n   The budget is a property of the corpus. Nothing about the")
    print("   identifiers changed between those rows.\n")

    print("2. u AS COLLISION ENTROPY, FROM A PUBLISHED FREQUENCY TABLE")
    print("   priced from data/, not measured from the graph\n")
    print(f"   {'field':<14}{'source':<34}{'u':>11}{'bits':>8}")
    print("   " + "-" * 67)
    for label, path, nc, fc, src in [
        ("surname", DATA / "surnames.csv", "name", "freq", "US Census surname file, top 2,000"),
        ("given_name", DATA / "givennames.csv", "name", "freq", "SSA given names, 1945-2005"),
    ]:
        u = u_from_frequency_table(path, nc, fc)
        print(f"   {label:<14}{src:<34}{u:>11.2e}{-math.log2(u):>8.1f}")

    # cross-check against what the pipeline measured
    try:
        w = {r["field"]: r for r in csv.DictReader(open(OUT / "weights.csv"))}
        print(f"\n   {'field':<14}{'from table':>12}{'measured':>11}{'gap':>8}")
        print("   " + "-" * 45)
        for label, path, nc, fc in [
            ("surname", DATA / "surnames.csv", "name", "freq"),
            ("given_name", DATA / "givennames.csv", "name", "freq"),
        ]:
            tb = -math.log2(u_from_frequency_table(path, nc, fc))
            ms = -math.log2(float(w[label]["u"]))
            print(f"   {label:<14}{tb:>12.1f}{ms:>11.1f}{ms-tb:>8.1f}")
        print("\n   Close, as it should be. Typos and nicknames spread values across")
        print("   more variants, which lowers u slightly and nudges the measured bits")
        print("   up. m falls at the same time, which is why log2(m/u) is the number")
        print("   that matters rather than either half on its own.")
    except (FileNotFoundError, KeyError):
        print("\n   (run the pipeline first to see the measured comparison)")

    # The ceiling quoted in the piece is for the whole US population, not for
    # the 2,000-name table the simulation draws from. The bundled shares are
    # each name's share of the whole population, so sum(p^2) over the top
    # 2,000 is a lower bound, and the names outside it (about half the
    # population, each below the 2,000th share) can add at most
    # (tail mass) x (largest tail share).
    lo, hi, cover, tail_max = surname_ceiling_bounds(DATA / "surnames.csv")
    print("\n   SURNAME CEILING FOR THE WHOLE POPULATION (the 10.8 bits in the piece)")
    print(f"   top 2,000 names cover {cover:.1%} of the population; sum(p^2) over them = {lo:.2e}")
    print(f"   names outside the top 2,000: {1-cover:.1%} of the population, each share < {tail_max:.1e},")
    print(f"   adding at most {hi-lo:.1e}; so sum(p^2) is between {lo:.2e} and {hi:.2e},")
    print(f"   and the ceiling -log2(u) is between {-math.log2(hi):.1f} and {-math.log2(lo):.1f} bits.")

    print("\n\n3. SAME HASH. DIFFERENT POPULATION. DIFFERENT EVIDENCE.")
    print("   one identifier, hashed identically, priced against five populations")
    print("   that differ only in how often one address covers two people\n")
    n = 2.5e8
    prior_bits, _, budget = size_budget(1e9, n)
    print(f"   {'shared':>8}{'u':>12}{'bits':>8}{'W':>9}{'precision':>12}{'clears?':>10}")
    print("   " + "-" * 60)
    for s in (0.005, 0.02, 0.05, 0.10, 0.25):
        u, bits = price_key(n, s)
        W = prior_bits + bits
        prec = 2 ** W / (1 + 2 ** W)
        print(f"   {s:>7.1%}{u:>12.2e}{bits:>8.1f}{W:>9.1f}{prec:>12.3%}"
              f"{'yes' if prec >= 0.99 else 'NO':>10}")
    print("\n   Identical hash function. Identical field. The evidence it carries")
    print("   moves by 5.7 bits, the precision it supports moves by 25 points,")
    print("   and the only thing that changed was household behaviour.")

    with open(OUT / "populations.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["share_rate", "u", "bits", "W", "precision"])
        for s in (0.005, 0.02, 0.05, 0.10, 0.25):
            u, bits = price_key(n, s)
            W = prior_bits + bits
            wr.writerow([s, f"{u:.4e}", f"{bits:.3f}", f"{W:.3f}",
                         f"{2**W/(1+2**W):.5f}"])


if __name__ == "__main__":
    main()
