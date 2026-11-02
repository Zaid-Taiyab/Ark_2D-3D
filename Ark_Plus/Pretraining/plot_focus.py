"""
plot_focus.py -- plot focused vs unfocused test mAUC from a run's *_focus.csv (written with --focus_eval True).
  python plot_focus.py RUN_DIR [--out PREFIX] [--no_student]
Focused = tested right after that dataset's own phase; unfocused = tested right after another dataset's phase.
Writes PREFIX_focus.png/.pdf (one panel per dataset + a panel with the focused - unfocused gap) and prints a table.
"""
import argparse, csv, glob, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("run_dir")
p.add_argument("--out", default="")
p.add_argument("--no_student", action="store_true")
a = p.parse_args()
files = glob.glob(os.path.join(a.run_dir, "Outputs", "*", "*_focus.csv")) + glob.glob(os.path.join(a.run_dir, "*_focus.csv"))
if not files:
    raise SystemExit(f"no *_focus.csv under {a.run_dir} (was the run started with --focus_eval True?)")
d = defaultdict(dict)   # (dataset, state) -> epoch -> (student, teacher)
for r in csv.DictReader(open(files[0])):
    d[(r["eval_dataset"], r["state"])][int(r["epoch"])] = (float(r["student_mAUC"]), float(r["teacher_mAUC"]))
names = sorted({k[0] for k in d})
out = a.out or os.path.join(a.run_dir, os.path.basename(os.path.normpath(a.run_dir)))
fig, axes = plt.subplots(1, len(names) + 1, figsize=(5.6 * (len(names) + 1), 4.4), squeeze=False)
rows = []
for k, ds in enumerate(names):
    ax = axes[0][k]
    for state, c in (("focused", "tab:green"), ("unfocused", "tab:orange")):
        e = sorted(d[(ds, state)])
        if not e:
            continue
        ax.plot(e, [d[(ds, state)][x][1] for x in e], "-o", ms=3, lw=2, color=c, label=f"{state} (teacher)")
        if not a.no_student:
            ax.plot(e, [d[(ds, state)][x][0] for x in e], "--", lw=1, color=c, alpha=0.7, label=f"{state} (student)")
    common = sorted(set(d[(ds, "focused")]) & set(d[(ds, "unfocused")]))
    gap_t = [d[(ds, "focused")][x][1] - d[(ds, "unfocused")][x][1] for x in common]
    axes[0][-1].plot(common, gap_t, "-o", ms=3, lw=2, label=f"{ds} (teacher)")
    if not a.no_student:
        gap_s = [d[(ds, "focused")][x][0] - d[(ds, "unfocused")][x][0] for x in common]
        axes[0][-1].plot(common, gap_s, "--", lw=1, label=f"{ds} (student)")
    if common:
        last = common[-1]
        rows.append((ds, last, d[(ds, "focused")][last], d[(ds, "unfocused")][last]))
    vals = [v for st in ("focused", "unfocused") for x in d[(ds, st)] if x > 0 for v in d[(ds, st)][x]]
    if vals:
        ax.set_ylim(max(0.5, min(vals) - 0.03), min(1.0, max(vals) + 0.02))
    ax.set_title(f"{ds}: test mAUC, focused vs unfocused", fontsize=10)
    ax.set_xlabel("epoch"); ax.set_ylabel("test mAUC"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
g = axes[0][-1]
g.axhline(0, color="grey", lw=1)
g.set_title("Gap: focused minus unfocused", fontsize=10); g.set_xlabel("epoch"); g.set_ylabel("mAUC difference")
g.grid(alpha=0.3); g.legend(fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{out}_focus.{ext}", dpi=160)
print(f"{'dataset':12s} {'epoch':>5s}  focused T/S        unfocused T/S      gap (teacher)")
for ds, e, (fs, ft), (us, ut) in rows:
    print(f"{ds:12s} {e:5d}  {ft:.4f} / {fs:.4f}    {ut:.4f} / {us:.4f}    {ft - ut:+.4f}")
print("saved:", f"{out}_focus.png", "(+ .pdf)")
