"""
Produce every asset in the published piece from pipeline output.

    python run_all.py --people 40000
    python make_blog_assets.py

Writes to out/blog/:
    fig1-signal-map.png        what each signal is worth
    fig2-stacking.png          adding signals is not adding up
    fig3-field-sets.png        declared vs device vs both
    table-fields.md            the per-signal table, as printed in the piece
    table-dependence.md        the three-row dependence table
    numbers.txt                every figure quoted in the body text, with its source

Nothing in this file is typed in. Every value is read from out/ or computed
from it, so if the pipeline changes, the assets change with it.
"""
import csv
import glob
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
BLOG = OUT / "blog"
BLOG.mkdir(parents=True, exist_ok=True)

PRETTY = {"given_name": "given name", "surname": "surname", "dob": "date of birth",
          "postal": "postal code", "phone": "phone", "email_hash": "hashed email",
          "ip": "IP address", "device_id": "device ID", "fpc": "first-party cookie",
          "tpc": "third-party cookie", "shared_id": "vendor ID"}
ENTITY = {"person": "a person", "family": "a family", "household": "a household",
          "device": "a device", "browser": "a browser"}
DECLARED = "given_name,surname,dob,postal,phone,email_hash"
DEVICE = "ip,device_id,fpc,tpc,shared_id"

sys.path.insert(0, str(ROOT / "src"))
from populations import surname_ceiling_bounds
SC_LO, SC_HI, _, _ = surname_ceiling_bounds(ROOT / "data" / "surnames.csv")

# ------------------------------------------------------------------ runs
def run_stacking(fields, out):
    cmd = [sys.executable, str(ROOT / "src" / "stacking.py"), "--out", out]
    if fields:
        cmd += ["--fields", fields]
    subprocess.run(cmd, check=True, capture_output=True)
    return list(csv.DictReader(open(OUT / out)))

print("running the three stacking configurations ...")
S_ALL = run_stacking("", "stacking_all.csv")
S_DEC = run_stacking(DECLARED, "stacking_declared.csv")
S_DEV = run_stacking(DEVICE, "stacking_device.csv")

W = [r for r in csv.DictReader(open(OUT / "weights.csv")) if r["field"] != "email"]
W.sort(key=lambda r: -float(r["expected_bits"]))
DEP = {}
for r in csv.DictReader(open(OUT / "dependence.csv")):
    DEP[(r["a"], r["b"])] = r
    DEP[(r["b"], r["a"])] = r
prior = float(open(OUT / "prior.txt").read())
TH = math.log2(99)
budget = -prior + TH
truth = list(csv.reader(open(OUT / "truth.csv")))[1:]
n_rec = len(truth)
n_ppl = len({t[1] for t in truth})

# ------------------------------------------------------------------ tables
with open(BLOG / "table-fields.md", "w") as f:
    f.write("| Signal | Identifies | Bits when it agrees | How often it agrees | Expected contribution |\n")
    f.write("|---|---|---:|---:|---:|\n")
    for r in W:
        f.write(f"| {PRETTY[r['field']]} | {ENTITY[r['entity']]} | {float(r['agree_bits']):.1f} "
                f"| {float(r['speaks_on_true_pairs'])*100:.0f}% | {float(r['expected_bits']):.1f} |\n")

pairs = [("surname + postal code", "surname", "postal"),
         ("hashed email + phone", "email_hash", "phone"),
         ("hashed email + vendor ID", "email_hash", "shared_id")]
with open(BLOG / "table-dependence.md", "w") as f:
    f.write("| Signal pair | Naive sum | Joint weight | Overcount |\n|---|---:|---:|---:|\n")
    for label, a, b in pairs:
        d = DEP[(a, b)]
        f.write(f"| {label} | {float(d['naive_bits']):.1f} | {float(d['joint_bits']):.1f} "
                f"| {abs(float(d['overcount_bits'])):.1f} |\n")

# ------------------------------------------------------------------ numbers in the text
w = {r["field"]: r for r in W}
ph, su = float(w["phone"]["agree_bits"]), float(w["surname"]["agree_bits"])
rec_all = float(S_ALL[-1]["recall"]); prec_all = float(S_ALL[-1]["precision"])
rec_dec = float(S_DEC[-1]["recall"]); prec_dec = float(S_DEC[-1]["precision"])
rec_dev = float(S_DEV[-1]["recall"]); prec_dev = float(S_DEV[-1]["precision"])
true_pairs = sum(c * (c - 1) // 2 for c in __import__("collections").Counter(t[1] for t in truth).values())
all_pairs = n_rec * (n_rec - 1) // 2
lines = [
 ("records", f"{n_rec:,}", "out/truth.csv"),
 ("people", f"{n_ppl:,}", "distinct person_id in out/truth.csv"),
 ("true same-person pairs", f"{true_pairs:,}", "sum of C(n,2) per person"),
 ("possible pairs", f"{all_pairs:,}", "C(records, 2)"),
 ("prior odds", f"{true_pairs/(all_pairs-true_pairs):.2e}", "true / (possible - true)"),
 ("prior, bits", f"{prior:.1f}", "out/prior.txt"),
 ("target for 99%, bits", f"{TH:.1f}", "log2(99)"),
 ("evidence budget, bits", f"{budget:.1f}", "-prior + log2(99)"),
 ("phone: bits when it agrees", f"{ph:.1f}", "out/weights.csv"),
 ("phone: m, u", f"{float(w['phone']['m']):.3f}, {float(w['phone']['u']):.2e}", "out/weights.csv"),
 ("surname: bits when it agrees", f"{su:.1f}", "out/weights.csv"),
 ("surname: m, u", f"{float(w['surname']['m']):.3f}, {float(w['surname']['u']):.2e}", "out/weights.csv"),
 ("surname: disagreement bits", f"{float(w['surname']['disagree_bits']):.1f}", "out/weights.csv"),
 ("phone: disagreement bits", f"{float(w['phone']['disagree_bits']):.1f}", "out/weights.csv"),
 ("phone alone: W, from displayed inputs", f"{prior:.1f} + {ph:.1f} = {round(prior,1)+round(ph,1):.1f}  (unrounded {prior+ph:.2f}; {'link' if prior+ph>=TH else 'no link'}, target {TH:.1f})", "equation (2); the piece shows the displayed-input sum"),
 ("phone + surname: W, from displayed inputs", f"{prior:.1f} + {ph:.1f} + {su:.1f} = {round(prior,1)+round(ph,1)+round(su,1):.1f}  (unrounded {prior+ph+su:.2f}; link)", "equation (2); the piece shows the displayed-input sum"),
 ("phone + surname: P(same person)", f"{2**(prior+ph+su)/(1+2**(prior+ph+su)):.2%}", "equation (6)"),
 ("phone + surname overlap, bits", f"{abs(float(DEP[('phone','surname')]['overcount_bits'])):.2f}", "out/dependence.csv"),
 ("first-party cookie: bits, availability", f"{float(w['fpc']['agree_bits']):.1f}, {float(w['fpc']['speaks_on_true_pairs']):.0%}", "out/weights.csv"),
 ("stacking: recall with phone alone", f"{float(S_ALL[0]['recall']):.3f}", "out/stacking_all.csv row 1"),
 ("stacking: recall after surname", f"{float(S_ALL[1]['recall']):.3f}", "out/stacking_all.csv row 2"),
 ("stacking: recall gained by last three", f"{float(S_ALL[-1]['recall'])-float(S_ALL[-4]['recall']):+.4f}", "out/stacking_all.csv"),
 ("all signals: recall, precision", f"{rec_all:.1%}, {prec_all:.3f}", "out/stacking_all.csv final row"),
 ("declared only: recall, precision", f"{rec_dec:.1%}, {prec_dec:.3f}", "out/stacking_declared.csv final row"),
 ("device only: recall, precision", f"{rec_dev:.1%}, {prec_dev:.3f}", "out/stacking_device.csv final row"),
 ("device signals add on top of declared, points", f"{(rec_all-rec_dec)*100:.1f}", "all minus declared"),
 ("surname ceiling, whole population: sum p^2", f"{SC_LO:.2e} to {SC_HI:.2e}", "raw Census shares in data/surnames.csv; src/populations.py section 2 shows the bound"),
 ("surname ceiling, whole population: bits", f"{-math.log2(SC_HI):.1f} to {-math.log2(SC_LO):.1f}", "-log2 of the line above"),
 ("cost ratio implied by the 99% threshold", f"{2**TH:.0f} : 1", "decision rule: link when odds exceed cost(false link) / cost(missed link); 99% confidence is 99:1 odds"),
]
with open(BLOG / "numbers.txt", "w") as f:
    f.write("Every number quoted in the piece, computed from out/\n\n")
    for k, v, src in lines:
        f.write(f"{k:<46} {v:<44} {src}\n")

# ------------------------------------------------------------------ figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
for fp in glob.glob(str(Path.home() / ".fonts" / "*.ttf")):
    fm.fontManager.addfont(fp)
FAM = "IBM Plex Sans" if any("Plex" in f.name for f in fm.fontManager.ttflist) else "DejaVu Sans"
WHITE, INK, GREY, GRID = "#FFFFFF", "#111111", "#5A5A5A", "#DCDCDC"
NAVY, BRICK, TEAL = "#1F3A63", "#9E2B25", "#2E6E6A"
plt.rcParams.update({"font.family": FAM, "figure.facecolor": WHITE, "axes.facecolor": WHITE,
                     "savefig.facecolor": WHITE, "text.color": INK, "axes.labelcolor": GREY,
                     "xtick.color": GREY, "ytick.color": GREY, "axes.edgecolor": GRID})
DECL = set(DECLARED.split(","))

def frame(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(1.1)
    ax.tick_params(length=4, width=1.0, labelsize=11.5)
    ax.grid(color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)

# fig 1 — signal map
fig, ax = plt.subplots(figsize=(10.5, 6.6), dpi=200)
fig.subplots_adjust(left=0.085, right=0.96, top=0.80, bottom=0.165)
fig.text(0.045, 0.935, "Strong is not the same as useful", fontsize=24, color=INK, weight=600)
fig.text(0.045, 0.885, "Each signal placed by how much it says when it agrees, and how often two records of the same person agree on it.",
         fontsize=12.5, color=GREY)
nudge = {"fpc": (2.0, 0), "tpc": (2.0, 0), "given_name": (1.6, -2.1), "surname": (1.6, 1.5), "postal": (1.6, -1.4), "ip": (1.6, 1.2)}
for r in W:
    x = float(r["speaks_on_true_pairs"]) * 100; y = float(r["agree_bits"]); e = float(r["expected_bits"])
    c = NAVY if r["field"] in DECL else TEAL
    ax.scatter(x, y, s=e * 60 + 40, color=c, alpha=0.85, zorder=5, edgecolor=WHITE, lw=1.5)
    dx, dy = nudge.get(r["field"], (1.6, 0))
    ax.text(x + dx, y + dy, PRETTY[r["field"]], fontsize=11.5, color=INK, va="center")
ax.text(4, 30.6, "says a lot, rarely available", fontsize=11.5, color=GREY, style="italic")
ax.text(46, 5.4, "says less, almost always available", fontsize=11.5, color=GREY, style="italic")
ax.set_xlim(0, 100); ax.set_ylim(4, 33)
ax.set_xlabel("how often two records of the same person agree on it", fontsize=12.5, labelpad=9)
ax.set_ylabel("evidence when it agrees, bits", fontsize=12.5, labelpad=9)
ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.set_xticklabels([f"{v}%" for v in (0, 20, 40, 60, 80, 100)])
frame(ax)
ax.scatter([], [], color=NAVY, s=50, label="declared by the person")
ax.scatter([], [], color=TEAL, s=50, label="observed from a device or browser")
ax.legend(frameon=False, fontsize=11.5, loc="upper right")
fig.text(0.045, 0.04, f"Circle size is expected contribution: evidence times availability. Synthetic data, {n_ppl:,} people.",
         fontsize=10.5, color=GREY)
fig.savefig(BLOG / "fig1-signal-map.png"); plt.close(fig)

# fig 2 — stacking
x = [int(r["n_fields"]) for r in S_ALL]; rec = [float(r["recall"]) for r in S_ALL]
pre = [float(r["precision"]) for r in S_ALL]; lab = [PRETTY[r["field_added"]] for r in S_ALL]
fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=200)
fig.subplots_adjust(left=0.085, right=0.96, top=0.80, bottom=0.24)
fig.text(0.045, 0.935, "Adding signals is not adding up", fontsize=24, color=INK, weight=600)
fig.text(0.045, 0.885, "Signals added one at a time, most useful first. The graph is rebuilt and scored against truth at every step.",
         fontsize=12.5, color=GREY)
ax.plot(x, pre, color=INK, lw=2.4, marker="o", ms=4.5, zorder=6, label="precision: of the links made, how many were right")
ax.plot(x, rec, color=NAVY, lw=2.4, marker="o", ms=4.5, zorder=6, label="recall: of the links that exist, how many were made")
ax.scatter([1], [rec[0]], s=150, facecolor=WHITE, edgecolor=BRICK, lw=2.2, zorder=8)
ax.annotate("the single most useful signal\nresolves nothing on its own", xy=(1.05, 0.03), xytext=(1.45, 0.86),
            fontsize=11.5, color=BRICK, linespacing=1.5, arrowprops=dict(arrowstyle="-", color=BRICK, lw=1.3))
ax.annotate(f"add one more and\n{rec[1]:.0%} resolves", xy=(2.05, rec[1] - 0.02), xytext=(3.0, 0.22),
            fontsize=11.5, color=NAVY, linespacing=1.5, arrowprops=dict(arrowstyle="-", color=NAVY, lw=1.3))
ax.annotate("the last three add\nalmost nothing", xy=(len(x) - 1, rec[-2] - 0.03), xytext=(len(x) - 3.7, 0.55),
            fontsize=11.5, color=GREY, linespacing=1.5, arrowprops=dict(arrowstyle="-", color=GREY, lw=1.1))
ax.set_xticks(x); ax.set_xticklabels(lab, rotation=38, ha="right", fontsize=11)
ax.set_ylim(-0.05, 1.08); ax.set_xlim(0.5, len(x) + 0.5); ax.set_yticks([0, .25, .5, .75, 1.0])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
frame(ax); ax.legend(frameon=False, fontsize=10.5, loc=(0.33, 0.08))
fig.text(0.045, 0.045, f"Synthetic data, {n_ppl:,} people, {n_rec:,} records. Precision stays near 100% because the method refuses links without enough evidence.",
         fontsize=10.5, color=GREY)
fig.savefig(BLOG / "fig2-stacking.png"); plt.close(fig)

# fig 3 — field sets, values computed from the three runs
fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=200)
fig.subplots_adjust(left=0.085, right=0.96, top=0.76, bottom=0.17)
fig.text(0.045, 0.92, "The two families work together", fontsize=24, color=INK, weight=600)
fig.text(0.045, 0.86, "Same synthetic population, same evidence threshold, three different sets of signals.", fontsize=12.5, color=GREY)
sets = ["device and browser\nsignals only", "declared attributes\nonly", "both families\ntogether"]
vals = [rec_dev * 100, rec_dec * 100, rec_all * 100]
bars = ax.bar(sets, vals, color=[TEAL, NAVY, INK], width=0.56, zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 2.2, f"{v:.1f}%", ha="center", fontsize=14, color=INK, weight=600)
ax.set_ylim(0, 110); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_ylabel("same-person record pairs recovered", fontsize=12.5, labelpad=9)
ax.tick_params(axis="x", labelsize=12)
frame(ax); ax.grid(axis="x", visible=False)
fig.text(0.045, 0.04, f"Precision was {prec_dev:.3f}, {prec_dec:.3f} and {prec_all:.3f} respectively. Declared attributes carry most of the resolution here; device signals add to it rather than replace it.",
         fontsize=10.5, color=GREY)
fig.savefig(BLOG / "fig3-field-sets.png"); plt.close(fig)

print("wrote:")
for p in sorted(BLOG.iterdir()):
    print("  ", p.relative_to(ROOT))
