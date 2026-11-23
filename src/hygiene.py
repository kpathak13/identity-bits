"""
Stage 1 — hygiene, including the case where the email arrives already hashed.

This is the part most write-ups skip. In practice a partner sends you 64 hex
characters, not an address, and almost everything hygiene normally does becomes
impossible:

  strip plus-addressing        needs the plaintext   -> impossible
  collapse gmail dots          needs the plaintext   -> impossible
  spot a disposable domain     needs the plaintext   -> impossible
  spot a relay/proxy address   needs the plaintext   -> impossible

Which is the sharper version of the argument about hashing. Hashing does not
change u. What it changes is your ability to raise m, because every
normalisation that would have raised it happens before you receive the value.
The same identifier is worth fewer bits hashed than in the clear — not because
the hash is weaker, but because you inherited someone else's hygiene.

Three things still work in hash space:

  1. CANONICALISE BEFORE HASHING, by agreement with the sender. Here 55% of
     senders apply a spec and 45% hash whatever the user typed. Both arrive
     looking identical, and they do not match each other.

  2. DENYLIST BY HASH. You cannot read a hash, but you can hash a known-bad
     value and compare. Junk addresses and role accounts are finite sets, so
     they are catchable.

  3. FREQUENCY. This is the general answer and it needs no plaintext at all. A
     hash appearing 8,000 times in 240,000 records is junk whatever it decodes
     to, and its value-level weight collapses accordingly. Stage 2 computes
     that, stage 3 uses it.

What is NOT catchable this way is a relay address: every one is unique, so it
is invisible to both denylists and frequency. It only shows up structurally, as
a key that never once agrees with anything.
"""
import csv
import hashlib
import re
from collections import Counter

from common import OUT, FIELDS


NICK_CANON = {
    "bob": "robert", "bill": "william", "rick": "richard", "jim": "james",
    "liz": "elizabeth", "kate": "katherine", "peggy": "margaret",
    "mike": "michael", "chris": "christopher", "jen": "jennifer",
    "pat": "patricia", "tom": "thomas", "joe": "joseph", "dan": "daniel",
    "matt": "matthew", "tony": "anthony", "charlie": "charles",
    "debbie": "deborah", "sue": "susan", "becky": "rebecca",
}
ROLE_LOCALS = ["info", "admin", "office", "contact", "sales", "billing",
               "support", "hello", "noreply"]
JUNK_EMAILS = ["test@test.com", "a@a.com", "none@none.com", "no@email.com",
               "asdf@asdf.com", "x@x.com", "test@gmail.com", "abc@abc.com",
               "nomail@nomail.com", "fake@fake.com"]
COMMON_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "icloud.com",
                  "aol.com", "comcast.net", "hotmail.com"]
DISPOSABLE = ["mailinator.com", "guerrillamail.com", "10minutemail.com",
              "tempmail.net", "throwaway.email"]


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def canon_email(s):
    s = (s or "").strip().lower()
    if "@" not in s:
        return s
    local, domain = s.split("@", 1)
    local = local.split("+", 1)[0]
    if domain in ("gmail.com", "googlemail.com"):
        local, domain = local.replace(".", ""), "gmail.com"
    return f"{local}@{domain}"


def build_denylist():
    """
    Every bad value we can enumerate, hashed. This is the whole trick: you
    cannot read a hash, but you can hash what you already suspect.
    """
    bad = set(JUNK_EMAILS)
    for role in ROLE_LOCALS:
        for dom in COMMON_DOMAINS + DISPOSABLE:
            bad.add(f"{role}@{dom}")
    out = set()
    for v in bad:
        out.add(sha(v))
        out.add(sha(canon_email(v)))
    return out


def norm_name(s):
    s = re.sub(r"[^a-z]", "", (s or "").strip().lower())
    return NICK_CANON.get(s, s)


def norm_postal(s):
    m = re.match(r"^(\d{5})", (s or "").strip())
    return m.group(1) if m else ""


def norm_phone(s):
    d = re.sub(r"\D", "", s or "")
    if len(d) == 11 and d.startswith("1"):
        d = d[1:]
    return d if len(d) == 10 else ""


def norm_dob(s):
    s = (s or "").strip()
    return s if re.match(r"^\d{4}-\d{2}-\d{2}$", s) else ""


def norm_ip(s):
    s = (s or "").strip()
    return s if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", s) else ""


def clean_plaintext_email(s):
    """Full hygiene. Only available when the address arrives in the clear."""
    s = (s or "").strip().lower()
    if "@" not in s:
        return "", None
    local, domain = s.split("@", 1)
    if s in JUNK_EMAILS:
        return "", "junk"
    if domain in DISPOSABLE:
        return "", "disposable"
    if domain == "privaterelay.appleid.com":
        return "", "relay"
    if local.split("+", 1)[0] in ROLE_LOCALS:
        return "", "role"
    return canon_email(s), None


def run(records):
    deny = build_denylist()
    hash_counts = Counter(r["email_sha256_canon"] for r in records
                          if r.get("email_sha256_canon"))
    n_hashed = sum(hash_counts.values())
    # a value held by more than 1 in 2,000 records is not one person's address
    FREQ_CUT = 1 / 2000
    frequent_bad = {h for h, c in hash_counts.items() if c / n_hashed > FREQ_CUT}

    st = Counter()
    clean = []
    for r in records:
        pt, reason = clean_plaintext_email(r.get("email"))
        if reason:
            st[f"plaintext dropped: {reason}"] += 1

        h = (r.get("email_sha256_canon") or "").lower()
        if h:
            if h in frequent_bad:
                st["hash dropped: too frequent to be a person"] += 1
                h = ""
            elif h in deny:
                st["hash dropped: matched a known-bad hash"] += 1
                h = ""

        if h and (r.get("email") or "").endswith("privaterelay.appleid.com"):
            st["RELAY SURVIVED IN HASH SPACE (uncatchable)"] += 1
        row = {
            "record_id": int(r["record_id"]),
            "source": r.get("source", ""),
            "given_name": norm_name(r.get("given_name")),
            "surname": norm_name(r.get("surname")),
            "dob": norm_dob(r.get("dob")),
            "postal": norm_postal(r.get("postal")),
            "email": pt,
            "email_hash": h,
            "phone": norm_phone(r.get("phone")),
            "ip": norm_ip(r.get("ip")),
        }
        for f in ("device_id", "fpc", "tpc", "shared_id"):
            row[f] = (r.get(f) or "").strip().upper()
        clean.append(row)
    return clean, st, deny, frequent_bad


def main():
    records = list(csv.DictReader(open(OUT / "records_raw.csv")))
    clean, st, deny, freq_bad = run(records)
    with open(OUT / "records_clean.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(clean[0].keys()))
        w.writeheader()
        w.writerows(clean)

    print("STAGE 1 — HYGIENE")
    print(f"  denylist holds {len(deny):,} hashes of known-bad addresses")
    print(f"  frequency flagged {len(freq_bad):,} more hashes as too common to be a person\n")
    for k in sorted(st):
        print(f"  {k:<34}{st[k]:>10,}")

    # the canonicalisation problem, measured
    diff = sum(1 for r in records
               if r.get("email_sha256") and
               r["email_sha256"] != r["email_sha256_canon"])
    tot = sum(1 for r in records if r.get("email_sha256"))
    print(f"\n  canonicalisation mismatch: {diff:,} of {tot:,} records ({diff/tot:.1%})")
    print("  hash the raw string and hash the canonical string and you get two")
    print("  different 64-character values for the same person. Those matches are")
    print("  lost silently — nothing in the data says a mismatch happened.")

    def fill(rows, k):
        return sum(1 for r in rows if r.get(k))
    print(f"\n  {'field':<14}{'before':>10}{'after':>10}")
    print("  " + "-" * 34)
    for k in FIELDS:
        b = fill(records, "email_sha256_canon") if k == "email_hash" else fill(records, k)
        print(f"  {k:<14}{b:>10,}{fill(clean,k):>10,}")


if __name__ == "__main__":
    main()
