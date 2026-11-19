"""
plot_focus_path.py -- focused / unfocused "chronological path" plots from a run's *_focus.csv
(written with --focus_eval True).

For each dataset X, at every tested epoch, X's test AUC is measured after EACH dataset's phase:
  focused   = right after X's own phase           (orange line with dots)
  unfocused = right after every other phase       (green crosses)
  path      = all points in the order they happened (dashed blue), so drops after other
              datasets' phases and the recovery at X's own phase show as a sawtooth.
Each tested epoch takes one unit on the x-axis; inside it, points sit at the phase positions
(within-epoch training order). One figure per model (student / teacher), one panel per dataset.

  python plot_focus_path.py RUN_DIR [--out PREFIX] [--models student teacher] [--datasets A B ...]
"""
import argparse
import csv
import glob
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {"VinDrCXR": "VinDr-CXR", "ChestXray14": "ChestX-ray14", "CTRATE": "CT-RATE"}

p = argparse.ArgumentParser()
p.add_argument("run_dir")
p.add_argument("--out", default="")
p.add_argument("--models", nargs="+", default=["student", "teacher"], choices=["student", "teacher"])
p.add_argument("--datasets", nargs="*", default=None)
p.add_argument("--title", default="")
a = p.parse_args()

files = glob.glob(os.path.join(a.run_dir, "Outputs", "*", "*_focus.csv")) + glob.glob(os.path.join(a.run_dir, "*_focus.csv"))
if not files:
    raise SystemExit(f"no *_focus.csv under {a.run_dir} (was the run started with --focus_eval True?)")

v, order = {}, []                                   # v[(epoch, after_phase, eval_ds)] = (student, teacher); later rows win
for r in csv.DictReader(open(files[0])):
    e = int(r["epoch"])
    v[(e, r["after_phase"], r["eval_dataset"])] = (float(r["student_mAUC"]), float(r["teacher_mAUC"]))
    if r["after_phase"] not in order:
        order.append(r["after_phase"])               # first appearance = training order within an epoch
epochs = sorted({k[0] for k in v})
datasets = a.datasets or [d for d in order if any(k[2] == d for k in v)]
N = len(order)
out = a.out or os.path.join(a.run_dir, os.path.basename(os.path.normpath(a.run_dir)))
order_txt = " -> ".join(NAMES.get(d, d) for d in order)

for model in a.models:
    mi = 0 if model == "student" else 1
    ncol = 2 if len(datasets) > 1 else 1
    nrow = (len(datasets) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.5 * ncol, 4.6 * nrow), squeeze=False)
    for idx, ds in enumerate(datasets):
        ax = axes[idx // ncol][idx % ncol]
        path_x, path_y, fx, fy, ux, uy, gaps = [], [], [], [], [], [], []
        for i, e in enumerate(epochs):
            foc, unf = None, []
            for k, ph in enumerate(order):
                if (e, ph, ds) not in v:
                    continue
                x, y = i + k / N, v[(e, ph, ds)][mi]
                path_x.append(x); path_y.append(y)
                if ph == ds:
                    fx.append(x); fy.append(y); foc = y
                else:
                    ux.append(x); uy.append(y); unf.append(y)
            if foc is not None and unf:
                gaps.append((e, foc - sum(unf) / len(unf)))
        ax.plot(path_x, path_y, "--", color="steelblue", lw=1.2, alpha=0.8, marker="x", ms=4,
                label="focus-unfocus chronological path", zorder=1)
        ax.plot(fx, fy, "-o", color="tab:orange", lw=2, ms=5, label="focused", zorder=3)
        ax.plot(ux, uy, "x", color="tab:green", ms=7, mew=1.6, label="unfocused", zorder=2)
        if fy:
            b = max(range(len(fy)), key=lambda j: fy[j])
            be = epochs[int(fx[b])]
            txt = f"Best focused: {fy[b]:.4f}\nEpoch {be} after {NAMES.get(ds, ds)}"
            if gaps:
                txt += f"\nGap (focused - mean unfocused):\nepoch {gaps[0][0]}: {gaps[0][1]:+.4f}  ->  epoch {gaps[-1][0]}: {gaps[-1][1]:+.4f}"
            ax.text(0.98, 0.04, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                    bbox=dict(boxstyle="round", fc="#dbe9f6", ec="steelblue"))
        vals = [y for y, x in zip(path_y, path_x) if x >= 1] or path_y
        if vals:
            lo, hi = min(vals), max(vals)
            pad = max(0.005, (hi - lo) * 0.12)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(range(len(epochs)))
        ax.set_xticklabels([str(e) for e in epochs])
        ax.set_xlim(-0.2, len(epochs) - 1 + (N - 1) / N + 0.2)
        ax.set_title(f"{NAMES.get(ds, ds)}: focused / unfocused ({model})", fontsize=11)
        ax.set_xlabel(f"Tested epoch (within-epoch order: {order_txt})", fontsize=9)
        ax.set_ylabel("test mAUC")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    for j in range(len(datasets), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    if a.title:
        fig.suptitle(a.title)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}_focuspath_{model}.{ext}", dpi=160)
    plt.close(fig)
    print("saved:", f"{out}_focuspath_{model}.png", "(+ .pdf)")
