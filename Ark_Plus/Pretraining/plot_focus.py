"""
plot_focus.py -- focused vs unfocused test mAUC from a run's *_focus.csv (written with --focus_eval True).

  python plot_focus.py RUN_DIR [--out PREFIX] [--zoom_from 50] [--title "..."]

2 x N grid: teacher on the top row, student on the bottom row, one column per dataset.
Focused = solid line, unfocused = dashed line. Two figures are written:
  PREFIX_focus.png / .pdf        the whole run (y-axis scaled from epoch 10)
  PREFIX_focus_zoom.png / .pdf   from --zoom_from onward, y-axis fitted tightly to the data
and a summary table is printed.
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {"VinDrCXR": "VinDr-CXR", "ChestXray14": "ChestX-ray14"}

p = argparse.ArgumentParser()
p.add_argument("run_dir")
p.add_argument("--out", default="")
p.add_argument("--zoom_from", type=int, default=50, help="first epoch of the zoomed figure")
p.add_argument("--title", default="Focused vs unfocused test mAUC")
a = p.parse_args()

files = glob.glob(os.path.join(a.run_dir, "Outputs", "*", "*_focus.csv")) + glob.glob(os.path.join(a.run_dir, "*_focus.csv"))
if not files:
    raise SystemExit(f"no *_focus.csv under {a.run_dir} (was the run started with --focus_eval True?)")
d = defaultdict(dict)   # (dataset, state) -> epoch -> {"student": x, "teacher": y}
for r in csv.DictReader(open(files[0])):
    d[(r["eval_dataset"], r["state"])][int(r["epoch"])] = {"student": float(r["student_mAUC"]),
                                                           "teacher": float(r["teacher_mAUC"])}
datasets = sorted({k[0] for k in d}, key=lambda s: (s != "VinDrCXR", s))
out = a.out or os.path.join(a.run_dir, os.path.basename(os.path.normpath(a.run_dir)))


def figure(first_epoch, zoom, suffix, title):
    fig, axes = plt.subplots(2, len(datasets), figsize=(5.5 * len(datasets), 8), squeeze=False)
    for r_i, who in enumerate(("teacher", "student")):
        for c_i, ds in enumerate(datasets):
            ax = axes[r_i][c_i]
            foc, unf = d[(ds, "focused")], d[(ds, "unfocused")]
            ef = [e for e in sorted(foc) if e >= first_epoch]
            eu = [e for e in sorted(unf) if e >= first_epoch]
            ax.plot(ef, [foc[e][who] for e in ef], "-", color="tab:blue", lw=1.8, label="focused")
            ax.plot(eu, [unf[e][who] for e in eu], "--", color="tab:orange", lw=1.5, label="unfocused")
            if zoom:
                vals = [foc[e][who] for e in ef] + [unf[e][who] for e in eu]
                if vals:
                    lo, hi = min(vals), max(vals)
                    pad = max(0.001, (hi - lo) * 0.15)
                    ax.set_ylim(lo - pad, hi + pad)
                    ax.set_xlim(first_epoch - 3, max(ef + eu) + 3)
            else:
                vals = [foc[e][who] for e in ef if e >= 10] + [unf[e][who] for e in eu if e >= 10]
                if vals:
                    ax.set_ylim(max(0.5, min(vals) - 0.02), min(1.0, max(vals) + 0.015))
            ax.set_title(f"{NAMES.get(ds, ds)}: {who}" + (f" (epochs {first_epoch}+)" if zoom else ""), fontsize=10)
            ax.set_xlabel("epoch")
            ax.set_ylabel("test mAUC")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="lower right")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}_{suffix}.{ext}", dpi=160)
    plt.close(fig)
    return f"{out}_{suffix}.png"


saved = [figure(0, False, "focus", a.title),
         figure(a.zoom_from, True, "focus_zoom", (a.title + f", zoomed (epochs {a.zoom_from}+)") if a.title else "")]

print(f"{'model':8s} {'dataset':12s} {'epoch':>5s}  {'focused':>8s}  {'unfocused':>9s}  mean difference (last test epochs)")
for who in ("teacher", "student"):
    for ds in datasets:
        foc, unf = d[(ds, "focused")], d[(ds, "unfocused")]
        common = sorted(set(foc) & set(unf))
        if common:
            late = common[-5:]
            g = sum(foc[e][who] - unf[e][who] for e in late) / len(late)
            print(f"{who:8s} {NAMES.get(ds, ds):12s} {common[-1]:5d}  {foc[common[-1]][who]:8.4f}  "
                  f"{unf[common[-1]][who]:9.4f}  {g:+.4f} (last {len(late)})")
for s in saved:
    print("saved:", s, "(+ .pdf)")
