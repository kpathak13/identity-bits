"""
Stage 2 — what each field is worth, in bits.

Per field, computed exactly by counting pairs rather than sampling. Collision
rates for a strong key sit near 1e-6, so a sampled estimate would be noise.

    u = P(agree | two records belong to DIFFERENT people)
    m = P(agree | two records belong to the SAME person)

    agreement weight    = log2(m / u)          positive
    disagreement weight = log2((1 - m) / (1 - u))   negative

Two extra columns matter once digital identifiers are in play.

ENTITY tells you what the field actually identifies. postal and ip identify a
HOUSEHOLD. A field can carry plenty of bits and still resolve the wrong thing:
being in the same home is not being the same person, and no amount of evidence
about the home closes that gap.

EXPECTED is agreement weight times the rate at which the field is both
populated and agreeing on true pairs. A cookie is worth a lot when it agrees
and almost never agrees, so its expected contribution is small.

The dependence check at the end is the part that matters for shared_id and
shared_id. It is derived from the email, so adding its weight to the
email's counts the same evidence two or three times.
"""
import csv
import math
import os
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

OFFLINE = ["given_name", "surname", "dob", "postal", "email", "email_hash", "phone"]
DIGITAL = ["ip", "device_id", "fpc", "tpc", "shared_id"]
FIELDS = OFFLINE + DIGITAL

# what the field actually identifies, regardless of how many bits it carries
ENTITY = {
    "given_name": "person", "surname": "family", "dob": "person",
    "postal": "household", "email": "person", "email_hash": "person",
    "phone": "person",
    "ip": "household", "device_id": "device", "fpc": "browser",
    "tpc": "browser", "shared_id": "person",
}

# every pair, so the scorer can subtract the measured overcount later
PAIRS_TO_CHECK = [(a, b) for i, a in enumerate(FIELDS) for b in FIELDS[i + 1:]]
NOTABLE = {("email", "email_hash"), ("email", "shared_id"),
           ("postal", "ip"), ("email", "phone"),
           ("surname", "postal"), ("dob", "surname"), ("device_id", "fpc")}


def c2(n):
    return n * (n - 1) // 2


def exact_mu(records, truth, field):
    pop = [r for r in records if r[field]]
    by_val, by_val_person, by_person = Counter(), Counter(), Counter()
    for r in pop:
        pid = truth[r["record_id"]]
        by_val[r[field]] += 1
        by_val_person[(r[field], pid)] += 1
        by_person[pid] += 1
    agree = sum(c2(n) for n in by_val.values())
    same_agree = sum(c2(n) for n in by_val_person.values())
    same_pairs = sum(c2(n) for n in by_person.values())
    cross_pairs = c2(len(pop)) - same_pairs
    m = same_agree / same_pairs if same_pairs else 0.0
    u = (agree - same_agree) / cross_pairs if cross_pairs else 0.0
    return m, u, same_pairs, cross_pairs, same_agree


def joint_u(records, truth, fa, fb):
    """u for 'both fields agree', restricted to pairs where both are populated."""
    pop = [r for r in records if r[fa] and r[fb]]
    by_val, by_val_person, by_person = Counter(), Counter(), Counter()
    for r in pop:
        pid = truth[r["record_id"]]
        key = (r[fa], r[fb])
        by_val[key] += 1
        by_val_person[(key, pid)] += 1
        by_person[pid] += 1
    agree = sum(c2(n) for n in by_val.values())
    same_agree = sum(c2(n) for n in by_val_person.values())
    same_pairs = sum(c2(n) for n in by_person.values())
    cross_pairs = c2(len(pop)) - same_pairs
    m_ab = same_agree / same_pairs if same_pairs else 0.0
    u_ab = (agree - same_agree) / cross_pairs if cross_pairs else 0.0
    return m_ab, u_ab, cross_pairs


def marginal_on(records, truth, field, restrict):
    """m and u for one field, on the subset where both fields are populated."""
    pop = [r for r in records if r[field] and r[restrict]]
    by_val, by_val_person, by_person = Counter(), Counter(), Counter()
    for r in pop:
        pid = truth[r["record_id"]]
        by_val[r[field]] += 1
        by_val_person[(r[field], pid)] += 1
        by_person[pid] += 1
    agree = sum(c2(n) for n in by_val.values())
    same_agree = sum(c2(n) for n in by_val_person.values())
    same_pairs = sum(c2(n) for n in by_person.values())
    cross_pairs = c2(len(pop)) - same_pairs
    m = same_agree / same_pairs if same_pairs else 0.0
    u = (agree - same_agree) / cross_pairs if cross_pairs else 0.0
    return m, u


def mu_on_subset(records, truth, field, keep):
    """m and u restricted to records from sources satisfying keep()."""
    sub = [r for r in records if keep(r) and r[field]]
    by_val, by_val_person, by_person = Counter(), Counter(), Counter()
    for r in sub:
        pid = truth[r["record_id"]]
        by_val[r[field]] += 1
        by_val_person[(r[field], pid)] += 1
        by_person[pid] += 1
    agree = sum(c2(n) for n in by_val.values())
    same_agree = sum(c2(n) for n in by_val_person.values())
    same_pairs = sum(c2(n) for n in by_person.values())
    cross = c2(len(sub)) - same_pairs
    if not same_pairs or not cross:
        return None
    m = same_agree / same_pairs
    u = max((agree - same_agree) / cross, 1.0 / cross)
    return m, u, math.log2(m / u), len(sub)


def prior_bits(records, truth):
    counts = Counter(truth[r["record_id"]] for r in records)
    true_pairs = sum(c2(c) for c in counts.values())
    all_pairs = c2(len(records))
    return math.log2(true_pairs / (all_pairs - true_pairs)), true_pairs, all_pairs


def main():
    records = list(csv.DictReader(open(OUT / "records_clean.csv")))
    for r in records:
        r["record_id"] = int(r["record_id"])
    truth = {int(a): int(b) for a, b in list(csv.reader(open(OUT / "truth.csv")))[1:]}

    p_bits, true_pairs, all_pairs = prior_bits(records, truth)
    rows = []
    for f in FIELDS:
        m, u, n_same, n_cross, same_agree = exact_mu(records, truth, f)
        u = max(u, 1.0 / n_cross)
        agree_bits = math.log2(m / u) if m > 0 else float("-inf")
        # how often does this field actually get to speak on a true pair?
        speak = same_agree / true_pairs
        rows.append({
            "field": f, "entity": ENTITY[f], "m": m, "u": u,
            "agree_bits": agree_bits,
            "disagree_bits": math.log2((1 - m) / (1 - u)) if m < 1 else float("nan"),
            "speaks_on_true_pairs": speak,
            "expected_bits": agree_bits * speak,
        })

    with open(OUT / "weights.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    open(OUT / "prior.txt", "w").write(str(p_bits))

    print("STAGE 2 — WHAT EACH FIELD IS WORTH")
    print(f"  true pairs {true_pairs:>13,}   all pairs {all_pairs:>17,}")
    print(f"  prior odds {2**p_bits:>13.3e}   = {p_bits:.1f} bits\n")
    print(f"  {'field':<12}{'entity':<11}{'m':>7}{'u':>11}{'agree':>8}"
          f"{'disagr':>8}{'speaks':>9}{'expect':>8}")
    print("  " + "-" * 72)
    for r in rows:
        d = "  n/a" if math.isnan(r["disagree_bits"]) else f"{r['disagree_bits']:.1f}"
        print(f"  {r['field']:<12}{r['entity']:<11}{r['m']:>7.3f}{r['u']:>11.2e}"
              f"{r['agree_bits']:>8.1f}{d:>8}{r['speaks_on_true_pairs']:>9.1%}"
              f"{r['expected_bits']:>8.1f}")
    need = -p_bits + math.log2(99)
    print(f"\n  budget for 99% precision: {-p_bits:.1f} + {math.log2(99):.1f} = {need:.1f} bits")

    print("\n  SAME COLUMN, DIFFERENT FIELD")
    print("  a date checked against a document and a date typed to clear an age")
    print("  gate are not the same variable. Split by source and they separate.\n")
    VERIFIED = {"crm", "partner_a"}
    print(f"  {'field':<12}{'source kind':<16}{'m':>7}{'u':>11}{'agree bits':>12}{'records':>10}")
    print("  " + "-" * 68)
    for f in ("dob", "email_hash", "postal"):
        for label, keep in (("verified", lambda r: r["source"] in VERIFIED),
                            ("self-asserted", lambda r: r["source"] not in VERIFIED)):
            got = mu_on_subset(records, truth, f, keep)
            if got:
                m_, u_, b_, n_ = got
                print(f"  {f:<12}{label:<16}{m_:>7.3f}{u_:>11.2e}{b_:>12.1f}{n_:>10,}")

    print("\n  VALUE-LEVEL WEIGHTS — what you can do without reading the hash")
    print("  a pair agreeing on value v is worth log2(m / p_v), not log2(m / u).")
    print("  frequency alone tells you a value is junk.\n")
    # computed on the PRE-hygiene hashes on purpose: dropping junk is one
    # option, downweighting it automatically is the other, and frequency does
    # that without anyone deciding what "junk" means.
    rawrows = list(csv.DictReader(open(OUT / "records_raw.csv")))
    vf = Counter(r["email_sha256_canon"] for r in rawrows if r["email_sha256_canon"])
    n_v = sum(vf.values())
    m_eh = next(r["m"] for r in rows if r["field"] == "email_hash")
    u_eh = next(r["u"] for r in rows if r["field"] == "email_hash")
    print(f"  {'rank':>5}{'count':>9}{'p_v':>11}{'value weight':>14}"
          f"{'field weight':>14}")
    print("  " + "-" * 53)
    for i, (v, c) in enumerate(vf.most_common(4), 1):
        p_v = c / n_v
        print(f"  {i:>5}{c:>9,}{p_v:>11.2e}{math.log2(m_eh / p_v):>14.1f}"
              f"{math.log2(m_eh / u_eh):>14.1f}")
    singles = [v for v, c in vf.items() if c == 1]
    rare = [c for c in vf.values() if c <= 2]
    p_rare = 2 / n_v
    print(f"  {'rare':>5}{2:>9,}{p_rare:>11.2e}{math.log2(m_eh / p_rare):>14.1f}"
          f"{math.log2(m_eh / u_eh):>14.1f}")
    print(f"\n  {len(singles):,} hashes appear exactly once. A value that never")
    print("  agrees with anything contributes nothing, whatever it is worth in theory.")
    with open(OUT / "value_weights.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["email_hash", "count", "p_v", "value_bits"])
        for v, c in vf.most_common(200):
            p_v = c / n_v
            w.writerow([v, c, f"{p_v:.6e}", f"{math.log2(m_eh / p_v):.3f}"])

    # the dependence matrix is 60+ field pairs, each needing three passes.
    # These rates are stable on a subsample, so cap the work on large corpora.
    dep_records = records
    cap = int(os.environ.get("DEP_SAMPLE", "60000"))
    if len(records) > cap:
        dep_records = random.Random(3).sample(records, cap)

    print("\n  DEPENDENCE — do these two fields carry separate evidence?")
    if dep_records is not records:
        print(f"  (estimated on a {cap:,}-record subsample)")
    print("  overcount is what the naive sum adds that the joint weight does not.\n")
    print(f"  {'field A':<12}{'field B':<12}{'naive sum':>11}{'joint':>9}"
          f"{'overcount':>11}")
    print("  " + "-" * 55)
    dep_rows = []
    for fa, fb in PAIRS_TO_CHECK:
        ma, ua = marginal_on(dep_records, truth, fa, fb)
        mb, ub = marginal_on(dep_records, truth, fb, fa)
        m_ab, u_ab, n_cross = joint_u(dep_records, truth, fa, fb)
        if min(ua, ub, m_ab) <= 0:
            continue
        u_ab = max(u_ab, 1.0 / n_cross)
        naive = math.log2(ma / ua) + math.log2(mb / ub)
        joint = math.log2(m_ab / u_ab)
        over = naive - joint
        dep_rows.append({"a": fa, "b": fb, "naive_bits": naive,
                         "joint_bits": joint, "overcount_bits": over})
        if (fa, fb) in NOTABLE:
            flag = "  <-- same evidence twice" if over > 4 else ""
            print(f"  {fa:<12}{fb:<12}{naive:>11.1f}{joint:>9.1f}{over:>11.1f}{flag}")
    with open(OUT / "dependence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dep_rows[0].keys()))
        w.writeheader()
        w.writerows(dep_rows)


if __name__ == "__main__":
    main()
