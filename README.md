# identity-bits

Prices identity links in **bits of evidence** and checks the arithmetic against
ground truth. Raw messy records, through hygiene, into a scored graph and a
pruned spine — eleven identifiers, offline and digital, including email that
arrives already hashed.

Everything below is reproduced by running the code. No data leaves your machine;
the population is synthetic and generated from public US frequency tables.

```bash
pip install -r requirements.txt
python run_all.py                    # 100k people, ~240k records, ~3 min
python run_all.py --people 20000     # smaller and faster
```

---

## The idea in one equation

```
W = log2(prior odds) + Σ log2(mᵢ / uᵢ)

m = P(field agrees | same person)      u = P(field agrees | different people)
agreement weight    = log2(m / u)
disagreement weight = log2((1 - m) / (1 - u))
```

m and u are **not properties of an identifier**. They are measurements of your
population and your sources. This repo measures them rather than assuming them.

---

## Reproducing the published piece exactly

The piece *Pricing identity links in bits of evidence* uses a 40,000-person
run. Two commands produce every table, figure and quoted number in it:

```bash
python run_all.py --people 40000
python make_blog_assets.py
```

`out/blog/` then contains the three figures as published, both tables in
Markdown, and `numbers.txt` listing every figure quoted in the body text with
the file it came from. Nothing in `make_blog_assets.py` is typed in; every
value is read from `out/` or computed from it.

The result tables further down this README are from the default
100,000-person run and differ from the piece in the third decimal.

---

## Result 1 — what each field is worth

`bits` is the weight when the field agrees. `speaks` is how often it agrees on a
true pair. `expected` is the product, and it reorders everything.

```
field       entity       bits  speaks  expected
phone       person       19.1     50%       9.6
surname     family        8.7     89%       7.8
ip          household    12.6     58%       7.3
postal      household    11.0     65%       7.2
given_name  person        7.8     89%       7.0
device_id   device       19.4     30%       5.9
email_hash  person       17.4     33%       5.8
dob         person        8.2     51%       4.2
fpc         browser      29.1     12%       3.6
shared_id   person       11.1     14%       1.6
tpc         browser      26.3      2%       0.5
```

The first-party cookie carries the most evidence of anything here and
contributes the third least, because two records of one person share a cookie
12% of the time. The third-party cookie has effectively stopped being an
identifier.

`entity` is fixed regardless of bits. `ip` and `postal` identify a household.
Being behind one router is not being one person.

---

## Result 2 — adding a field is not the same as ranking it

Fields added by expected contribution, spine rebuilt and rescored each step:

```
+ field        cum bits    prec  recall     F1     ΔF1
phone               9.6   1.000   0.000  0.000  +0.000
surname            17.3   1.000   0.446  0.617  +0.617
ip                 24.7   1.000   0.486  0.654  +0.037
postal             31.8   0.998   0.692  0.817  +0.164
given_name         38.8   1.000   0.892  0.943  +0.126
device_id          44.7   1.000   0.927  0.962  +0.019
email_hash         50.6   1.000   0.942  0.970  +0.008
dob                54.7   0.999   0.969  0.984  +0.014
fpc                58.3   0.999   0.975  0.987  +0.003
shared_id          59.9   0.999   0.976  0.987  +0.000
tpc                60.3   0.999   0.976  0.988  +0.000
```

The biggest contributor resolves **nothing** on its own — 19.1 bits, and not one
pair clears the line. A second field worth 8.7 bits resolves 45%, because two
fields can now clear it together. The budget belongs to the link.

---

## Result 3 — probabilistic vs deterministic

```
spine                          precision   recall     F1
deterministic, email hash          0.793    0.334  0.470
deterministic, email OR phone      0.759    0.737  0.748
evidence budget                    0.999    0.976  0.988
```

Not a trade-off. The budget wins on both axes at once.

---

## Result 4 — deterministic cannot be ground truth

Close on email-hash equality and use it as labels — what teams mean by "our
deterministic matches are the truth set". Those labels are themselves wrong on
21% of the pairs they assert and miss 67% of real ones. Worse, they are wrong
in a direction:

```
spine                          vs real truth        vs det. labels
                            prec recall     F1    prec recall     F1
deterministic, email hash  0.793  0.334  0.470   1.000  1.000  1.000
evidence budget            0.999  0.976  0.988   0.337  0.779  0.471
```

The labels invert the ranking. The best spine scores worst, and the
deterministic spine scores a perfect 1.000 because it is being graded by the
rule that produced it. 2,752 clusters where the labels say "one person", truth
disagrees, and the budget correctly split them are all counted as errors.

---

## Result 5 — hashing takes away m, not u

Most of hygiene becomes impossible once the address is a hash: no stripping
plus-addressing, no collapsing gmail dots, no spotting a disposable domain or a
relay address. m drops from 0.812 to 0.699 for the same underlying addresses.

Three things still work:

1. **Canonicalise before hashing, by agreement.** 7.8% of records here have
   `sha256(raw) != sha256(canonical)`. Both are 64 hex characters, they never
   match, and nothing says a mismatch happened.
2. **Denylist by hash** — you cannot read a hash, but you can hash what you
   suspect. 118 entries cover junk and role accounts.
3. **Frequency**, which needs no plaintext. A hash appearing 919 times is not
   one person's address. Value weight `log2(m / p_v)` drops it from 18.7 bits to
   6.9 automatically.

Relay addresses defeat all three: each is unique, so no frequency signature and
nothing to denylist. 11,209 survive untouched.

---

## Result 6 — the arithmetic predicts reality

A pair carrying W bits should be a true match with probability
`2^W / (1 + 2^W)`:

```
    bits     pairs   predicted   observed     gap
   -19.9   213,132       0.000      0.000   0.000
    -8.2   129,559       0.003      0.006   0.002
     3.5    11,654       0.921      0.900  -0.021
    15.3    40,705       1.000      1.000  -0.000
```

Within two points across twenty orders of magnitude in odds. Before the
dependence correction the gap at the crossover reaches 0.48 — calibration is how
you detect correlated evidence you did not know to look for.

---

## Dependence

```
field A     field B       naive sum    joint  overcount
email       email_hash         37.5     18.8       18.8
email       shared_id          36.6     18.3       18.3
postal      ip                 24.0     15.7        8.3
email       phone              38.9     31.1        7.8
surname     postal             19.9     19.9        0.0
dob         surname            23.0     23.3       -0.3
```

`shared_id` is derived from the email, so agreeing on both is agreeing once.
`email`/`email_hash` is the control — literally the same value, and the measured
overcount is the field's entire weight. `surname`/`postal` at zero is the other
control.

The scorer subtracts these greedily: strongest agreeing field counted whole,
each later one charged its largest overlap with a field already counted.
Subtracting every pair removes k(k−1)/2 terms and buries true matches; greedy
removes k−1 and reproduces the measured joint weights.

---

## Stages

| stage | file | produces |
|---|---|---|
| 0 | `src/generate.py` | records with ground truth and realistic defects |
| 1 | `src/hygiene.py` | normalised records, plaintext and hash paths |
| 2 | `src/weights.py` | m, u, bits, per-source split, dependence matrix |
| 3 | `src/graph.py` | candidate pairs scored two ways |
| 4 | `src/spine.py` | eleven spine strategies compared |
| 5 | `src/evaluate.py` | calibration, threshold sweep, charts |
| 6 | `src/stacking.py` | what each added field buys |
| 7 | `src/groundtruth.py` | the two claims above, tested |
| 8 | `src/populations.py` | the post's worked example, and one hash across five populations |

---

## Result 7 — same hash, different population, different evidence

The rest of the repo measures one population. This prices one identifier,
hashed identically, against five that differ only in how often an address
covers more than one person:

```
  shared           u    bits        W   precision   clears 99%?
   0.5%    2.00e-11    35.5      7.2     99.303%       yes
   2.0%    8.00e-11    33.5      5.2     97.270%        no
   5.0%    2.00e-10    32.1      3.8     93.443%        no
  10.0%    4.00e-10    31.1      2.8     87.692%        no
  25.0%    1.00e-09    29.8      1.5     74.026%        no
```

Same hash function, same field, 5.7 bits and 25 points of precision apart.

The same stage reproduces the post's worked example — 1B records over 250M
people gives -28.3 bits and a 34.9 bit budget — using the function that sizes
every other corpus in this repo, and prices `surname` and `given_name` straight
from the bundled Census and SSA frequency tables as `u = sum(p_i^2)`. Those come
out at 8.8 and 7.9 bits against 8.9 and 8.0 measured from the graph, which is
the check that the empirical and analytic routes agree.

---

## Where every number comes from

Three kinds of source, and they should not be confused. None is a measurement
from any production system.

| source | what it sets | examples |
|---|---|---|
| **Public data** | how often two strangers share a name | surname and given-name collision rates, from Census 2000 and SSA frequency tables bundled in `data/` |
| **Arithmetic** | the baseline uniqueness of a token | a random 48-bit cookie or device ID collides with probability about 2⁻⁴⁸; a random 10-digit phone number or email address almost never collides by chance |
| **Chosen parameters** | every rate that makes an identifier imperfect | all of the table below |

Only the cookies keep a purely arithmetic collision rate, because no sharing is
injected on them. For every other identifier the measured u is driven by the
parameters below, and every m and availability figure is too.

Two numbers in the piece are arithmetic on public data rather than pipeline
output, and `make_blog_assets.py` prints both at the end of `numbers.txt`:

- The surname ceiling of about 10.8 bits is for the whole US population. The
  shares in `data/surnames.csv` are each name's share of the whole population
  (the top 2,000 cover 48.9%), so Σp² over them is a lower bound; the names
  outside the top 2,000 each have a share below the 2,000th and can add at most
  3.1 × 10⁻⁵, which puts Σp² between 5.55 × 10⁻⁴ and 5.87 × 10⁻⁴ and the
  ceiling between 10.7 and 10.8 bits. `src/populations.py` prints the bound.
- The "roughly a hundred times" in the piece is the cost ratio implied by the
  99 percent threshold: the decision rule links a pair when its odds of being
  a match exceed cost(false link) / cost(missed link), and 99 percent
  confidence is 99-to-1 odds.

### Parameters, as set in `src/generate.py`

| what | value | affects |
|---|---|---|
| records per person | 1–6, weighted .30/.30/.20/.15/.05 | prior, all m |
| household size | 1–4, weighted .28/.34/.22/.16 | postal, IP, device, email sharing |
| distinct postal codes | 3,000, households assigned uniformly | postal u (a crude choice; real ZIP populations are skewed) |
| people who move (second postal on later records) | 18% | postal m |
| people sharing a household email | 12% | email u |
| people using a household device | 9% of people; 35% of their records | device u |
| recycled phone numbers | 3% | phone u |
| email changed on later records | 16% | email m |
| phone changed on later records | 13% | phone m |
| nickname used | 10% | given-name m |
| typo in given name / surname | 6% / 6% | name m |
| role account (`info@`) | 4% | email u, hygiene |
| junk email (`test@test.com`) | 4.5% | email u, hygiene |
| disposable domain | 2.5% | hygiene |
| relay address (`privaterelay`) | 9% | email m |
| plus-addressing / Gmail dot variants | 5% / 7% | email m before canonicalisation |
| sender canonicalised before hashing | 55% | hashed-email m |
| hashed-only records (no plaintext) | 35% | hygiene coverage |
| IP: carrier NAT pool | 7% of records, 60 shared addresses | IP u |
| IP: lease rotation on later records | 28% | IP m |
| device ID reset | 20% | device m |
| browsers per person / cookie churn | 1–3 / 40% of records get a fresh cookie | 1P-cookie availability |
| third-party cookie present at all | 35%; 55% of those are fresh | 3P-cookie availability |
| vendor ID present | 55% of records with an email | vendor-ID availability |
| DOB falsified at unverified sources | 34%; 55% of fakes are sentinel dates | DOB m and u |
| DOB off-by-one-year error | 2% verified, 4% unverified | DOB m |
| missing: DOB / postal / email / phone / IP / device / 1P cookie | 12 / 8 / 15 / 22 / 10 / 30 / 25% | availability |

These were chosen to be plausible, meaning a practitioner would not find them
absurd. They are not sourced to a study and should not be quoted as industry
rates. The phenomena themselves — household email sharing, recycled numbers,
cookie churn, age-gate date falsification — are widely described in public
discussion of identity data; the rates are judgment.

---

## Limits worth stating

**The population is simulated and the defect rates are chosen.** Household email
sharing at 12%, DOB falsification at 34% of unverified sources, cookie churn at
40% — these are plausible, not measured. What generalises is the mechanism and
the method, not the specific numbers.

**The rankings move when the simulation changes.** Modelling DOB falsification
moved DOB from first to eighth. That is the method working, not the method
failing, and it is the reason the repo exists: measure your own data rather than
inheriting a table.

**m needs labels.** Here it comes from ground truth. In production you would
estimate it with EM, which is itself an unsupervised model.

---

## Knobs

- `--people` — corpus size, which moves the prior and the budget
- `--fields` on `stacking.py` — restrict the experiment to a subset, e.g. a
  digital-only graph with no name or DOB
- household sharing, relay rate, junk rate, canonicalisation share in
  `generate.py`
- `FREQ_CUT` in `hygiene.py`, `MAX_BLOCK` in `graph.py`, `MAX_PLAUSIBLE` in
  `spine.py`
