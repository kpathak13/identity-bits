"""
Stage 0 — synthetic population with ground truth.

People are drawn from real US frequency distributions, grouped into households,
then emitted as messy records the way source systems actually emit them.

The defects are the point. Each one is chosen because it breaks a different
assumption a matching rule quietly makes:

  offline
    typos, nicknames, casing            m falls
    missing fields                      coverage falls
    people move (second postal)         m falls for postal
    email and phone change              m falls for the strongest keys
    12% of people share a household email    u rises for email
    3% inherit a recycled phone              u rises for phone
    role accounts (info@, admin@)            u rises catastrophically

  digital
    IP is a HOUSEHOLD address, not a person's    wrong entity
    IP leases rotate                             m falls
    7% of traffic sits behind carrier NAT        u rises hard
    device IDs get reset                         m falls
    9% of people use a household device          u rises
    first-party cookies churn and are per-browser    m very low
    third-party cookies are mostly blocked           coverage very low
    shared_id is derived from the email           perfectly correlated

Every record carries person_id. Only evaluate.py may look at it.
"""
import csv
import hashlib
import os
import random
import string
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

NICKNAMES = {
    "Robert": "Bob", "William": "Bill", "Richard": "Rick", "James": "Jim",
    "Elizabeth": "Liz", "Katherine": "Kate", "Margaret": "Peggy",
    "Michael": "Mike", "Christopher": "Chris", "Jennifer": "Jen",
    "Patricia": "Pat", "Thomas": "Tom", "Joseph": "Joe", "Daniel": "Dan",
    "Matthew": "Matt", "Anthony": "Tony", "Charles": "Charlie",
    "Deborah": "Debbie", "Susan": "Sue", "Rebecca": "Becky",
}
DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "icloud.com", "aol.com",
           "comcast.net", "hotmail.com"]
ROLE_LOCALS = ["info", "admin", "office", "contact", "sales", "billing"]
# values people type to get past a form. A handful of strings, millions of rows.
JUNK_EMAILS = ["test@test.com", "a@a.com", "none@none.com", "no@email.com",
               "asdf@asdf.com", "x@x.com", "test@gmail.com", "abc@abc.com",
               "nomail@nomail.com", "fake@fake.com"]
DISPOSABLE = ["mailinator.com", "guerrillamail.com", "10minutemail.com",
              "tempmail.net", "throwaway.email"]
# What people type to clear an age gate, and what systems write when the user
# left the field blank. A handful of strings covering a large share of rows.
SENTINEL_DOBS = ["1970-01-01", "1900-01-01", "1980-01-01", "1990-01-01",
                 "2000-01-01", "1111-11-11", "1969-12-31"]
# sources that check the date against a document, and sources that do not
VERIFIED_SOURCES = {"crm", "partner_a"}

FIELDNAMES = ["record_id", "source", "given_name", "surname", "dob", "postal",
              "email", "email_sha256", "email_sha256_canon", "phone", "ip",
              "device_id", "fpc", "tpc", "shared_id"]


def _load_weighted(path, name_col, freq_col):
    rows = list(csv.DictReader(open(path)))
    names = [r[name_col] for r in rows]
    freqs = [float(r[freq_col]) for r in rows]
    t = sum(freqs)
    return names, [f / t for f in freqs]


def _typo(s, rng):
    if len(s) < 4:
        return s
    kind = rng.choice(["swap", "drop", "double", "sub"])
    i = rng.randrange(1, len(s) - 1)
    if kind == "swap":
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if kind == "drop":
        return s[:i] + s[i + 1:]
    if kind == "double":
        return s[:i] + s[i] + s[i:]
    return s[:i] + rng.choice(string.ascii_lowercase) + s[i + 1:]


def _sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def _canon(s):
    """The canonicalisation a sender SHOULD apply before hashing."""
    s = s.strip().lower()
    if "@" not in s:
        return s
    local, domain = s.split("@", 1)
    local = local.split("+", 1)[0]
    if domain in ("gmail.com", "googlemail.com"):
        local, domain = local.replace(".", ""), "gmail.com"
    return f"{local}@{domain}"


def _id(rng, prefix):
    return f"{prefix}-{rng.getrandbits(48):012x}"


def build(n_people=100000, seed=7):
    rng = random.Random(seed)
    sur, sur_p = _load_weighted(DATA / "surnames.csv", "name", "freq")
    giv, giv_p = _load_weighted(DATA / "givennames.csv", "name", "freq")
    zips = [f"{z:05d}" for z in rng.sample(range(1000, 99999), 3000)]

    # ---- people -------------------------------------------------------
    people = []
    for pid in range(n_people):
        g = rng.choices(giv, giv_p)[0]
        s = rng.choices(sur, sur_p)[0]
        people.append({
            "person_id": pid, "given": g, "surname": s,
            "dob": f"{rng.randint(1945,2005)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            "email": f"{g.lower()}.{s.lower()}{rng.randint(1,999)}@{rng.choice(DOMAINS)}",
            "email2": f"{g.lower()[0]}{s.lower()}{rng.randint(1,999)}@{rng.choice(DOMAINS)}",
            # the fake date this person reaches for when a form demands one.
            # They reuse it, but it is not their real date.
            "dob_fake": (rng.choice(SENTINEL_DOBS) if rng.random() < 0.55 else
                         f"{rng.randint(1960,2000)}-{rng.randint(1,12):02d}-"
                         f"{rng.randint(1,28):02d}"),
            "phone": f"{rng.randint(200,989)}{rng.randint(200,999)}{rng.randint(1000,9999)}",
            "phone2": f"{rng.randint(200,989)}{rng.randint(200,999)}{rng.randint(1000,9999)}",
            "device": _id(rng, "D"),
            "device_shared": None,
            "fpcs": [_id(rng, "F") for _ in range(rng.choice([1, 2, 3]))],
            "tpcs": [_id(rng, "T") for _ in range(rng.choice([1, 2]))],
        })

    # ---- households ---------------------------------------------------
    # A home shares a postal code and an IP. This is why those two are
    # household-level evidence even when they look person-level.
    hh_of, cur, left = [], 0, 0
    for _ in range(n_people):
        if left == 0:
            cur += 1
            left = rng.choices([1, 2, 3, 4], [.28, .34, .22, .16])[0]
        hh_of.append(cur)
        left -= 1
    n_hh = cur

    hh_zip = {h: rng.choice(zips) for h in range(1, n_hh + 1)}
    hh_ip = {h: f"{rng.randint(24,203)}.{rng.randint(0,255)}."
                f"{rng.randint(0,255)}.{rng.randint(1,254)}"
             for h in range(1, n_hh + 1)}
    cgnat = [f"100.{rng.randint(64,127)}.{rng.randint(0,255)}.{rng.randint(1,254)}"
             for _ in range(60)]

    for p in people:
        h = hh_of[p["person_id"]]
        p["hh"] = h
        p["zip"] = hh_zip[h]
        p["zip2"] = rng.choice(zips)
        p["ip"] = hh_ip[h]
        p["ip2"] = (hh_ip[h] if rng.random() < 0.5 else
                    f"{rng.randint(24,203)}.{rng.randint(0,255)}."
                    f"{rng.randint(0,255)}.{rng.randint(1,254)}")

    by_hh = {}
    for p in people:
        by_hh.setdefault(p["hh"], []).append(p)

    # 9% of people mostly use a household member's device
    shared_dev = 0
    for p in people:
        mates = [q for q in by_hh[p["hh"]] if q is not p]
        if mates and rng.random() < 0.09:
            p["device_shared"] = rng.choice(mates)["device"]
            shared_dev += 1

    # 12% of people share a household email
    shared_email = 0
    for p in people:
        mates = [q for q in by_hh[p["hh"]] if q is not p]
        if mates and rng.random() < 0.12:
            p["email"] = rng.choice(mates)["email"]
            shared_email += 1

    # 3% inherit a recycled phone number from an unrelated earlier person
    recycled = 0
    for i in range(1, n_people):
        if rng.random() < 0.03:
            people[i]["phone"] = people[rng.randrange(i)]["phone"]
            recycled += 1

    # ---- records ------------------------------------------------------
    records, truth = [], {}
    rid = 0
    for p in people:
        n_rec = rng.choices([1, 2, 3, 4, 6], [.30, .30, .20, .15, .05])[0]
        for k in range(n_rec):
            given, surname = p["given"], p["surname"]
            if rng.random() < 0.10:
                given = NICKNAMES.get(given, given)
            if rng.random() < 0.06:
                given = _typo(given.lower(), rng).title()
            if rng.random() < 0.06:
                surname = _typo(surname.lower(), rng).title()

            zipc = p["zip2"] if (k > 0 and rng.random() < 0.18) else p["zip"]
            email = p["email2"] if (k > 0 and rng.random() < 0.16) else p["email"]
            phone_v = p["phone2"] if (k > 0 and rng.random() < 0.13) else p["phone"]

            if rng.random() < 0.04:
                email = f"{rng.choice(ROLE_LOCALS)}@{email.split('@')[1]}"
            if rng.random() < 0.045:
                email = rng.choice(JUNK_EMAILS)           # typed to skip the form
            elif rng.random() < 0.025:
                email = f"{given.lower()}{rng.randint(1,9999)}@{rng.choice(DISPOSABLE)}"
            elif rng.random() < 0.09:
                # relay address: unique per person per app, never shared, and it
                # rotates. Looks like a perfect key and behaves like a cookie.
                email = f"{rng.getrandbits(64):016x}@privaterelay.appleid.com"
            if rng.random() < 0.05:
                lo, dm = email.split("@")
                email = f"{lo}+shop@{dm}"
            if rng.random() < 0.07 and "gmail" in email:
                lo, dm = email.split("@")
                email = f"{lo.replace('.', '')}@{dm}"

            ip = (rng.choice(cgnat) if rng.random() < 0.07 else
                  (p["ip2"] if (k > 0 and rng.random() < 0.28) else p["ip"]))

            device = p["device"]
            if p["device_shared"] and rng.random() < 0.35:
                device = p["device_shared"]
            if rng.random() < 0.20:
                device = _id(rng, "D")               # advertising ID reset

            fpc = rng.choice(p["fpcs"]) if rng.random() > 0.40 else _id(rng, "F")
            tpc = ""
            if rng.random() < 0.35:                  # survives blocking
                tpc = rng.choice(p["tpcs"]) if rng.random() > 0.55 else _id(rng, "T")

            # shared_id follows the email used at this touchpoint, so it
            # inherits every defect the email has.
            shared_id = ("S-%012d" % (int(_sha(email), 16) % 10**12)) if rng.random() < 0.55 else ""

            src = rng.choice(["crm", "web", "partner_a", "partner_b"])
            # A verified source checked the date against a document. An
            # unverified one accepted whatever cleared the age gate, and the
            # same person does not always type the same lie.
            if src in VERIFIED_SOURCES:
                dob_v = p["dob"] if rng.random() > 0.02 else \
                    f"{int(p['dob'][:4]) + rng.choice([-1, 1])}{p['dob'][4:]}"
            elif rng.random() < 0.34:
                dob_v = (p["dob_fake"] if rng.random() < 0.6
                         else rng.choice(SENTINEL_DOBS))
            else:
                dob_v = p["dob"] if rng.random() > 0.04 else \
                    f"{int(p['dob'][:4]) + rng.choice([-1, 1])}{p['dob'][4:]}"

            rec = {
                "record_id": rid,
                "source": src,
                "given_name": given, "surname": surname,
                "dob": dob_v if rng.random() > 0.12 else "",
                "postal": zipc if rng.random() > 0.08 else "",
                "email": email if rng.random() > 0.15 else "",
                # what a partner actually sends. Some canonicalise first, some
                # hash whatever the user typed. Both arrive as 64 hex chars and
                # look identical.
                "email_sha256": "", "email_sha256_canon": "",
                "phone": phone_v if rng.random() > 0.22 else "",
                "ip": ip if rng.random() > 0.10 else "",
                "device_id": device if rng.random() > 0.30 else "",
                "fpc": fpc if rng.random() > 0.25 else "",
                "tpc": tpc,
                "shared_id": shared_id,
            }
            if rng.random() < 0.3:
                rec["given_name"] = rec["given_name"].upper()
                rec["surname"] = rec["surname"].upper()
            if rec["phone"] and rng.random() < 0.4:
                d = rec["phone"]
                rec["phone"] = f"({d[:3]}) {d[3:6]}-{d[6:]}"
            if rec["postal"] and rng.random() < 0.15:
                rec["postal"] += "-" + f"{rng.randint(1000,9999)}"
            if rec["email"]:
                raw = rec["email"]
                rec["email_sha256"] = _sha(raw)
                # ~55% of senders apply a canonicalisation spec before hashing
                rec["email_sha256_canon"] = _sha(_canon(raw)) if rng.random() < 0.55 \
                    else _sha(raw)
                if rng.random() < 0.35:
                    rec["email"] = ""      # many partners send the hash only
            records.append(rec)
            truth[rid] = p["person_id"]
            rid += 1

    stats = {
        "people": n_people,
        "households": n_hh,
        "records": len(records),
        "people sharing a household email": shared_email,
        "people using a household device": shared_dev,
        "people with a recycled phone": recycled,
    }
    return records, truth, stats


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int,
                    default=int(os.environ.get("N_PEOPLE", "100000")),
                    help="number of distinct people to simulate")
    a = ap.parse_args()
    records, truth, stats = build(n_people=a.people)
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    with open(out / "records_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(records)
    with open(out / "truth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["record_id", "person_id"])
        w.writerows(sorted(truth.items()))
    print("STAGE 0 — GENERATE")
    for k, v in stats.items():
        print(f"  {k:<34} {v:>10,}")
    fill = lambda k: sum(1 for r in records if r[k])
    print("  coverage")
    for k in FIELDNAMES[2:]:
        print(f"    {k:<14}{fill(k):>8,}  {fill(k)/len(records):>6.1%}")


if __name__ == "__main__":
    main()
