"""
plot_lr_finder.py  --  plot one or more lr_finder_<dataset>.csv files (loss vs learning rate, log scale)
and mark the suggestions.  Usage:  python plot_lr_finder.py lr_finder_*.csv  [--out folder]
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lr_finder import smooth, suggest  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("csv", nargs="+")
p.add_argument("--out", default=".")
a = p.parse_args()
n = len(a.csv)
fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), squeeze=False)
rows = []
for ax, path in zip(axes[0], a.csv):
    name = os.path.basename(path).replace("lr_finder_", "").replace(".csv", "")
    lr, loss = [], []
    for r in csv.DictReader(open(path)):
        lr.append(float(r["lr"])); loss.append(float(r["loss"]))
    s = smooth(loss)
    sug = suggest(lr, loss)
    ax.plot(lr, loss, color="tab:blue", alpha=0.25, lw=1, label="loss per step")
    ax.plot(lr, s, color="tab:blue", lw=2, label="smoothed")
    if sug:
        ax.axvline(sug["min_loss_div10"], color="tab:green", ls="--", lw=1.5, label=f"suggested peak ~{sug['min_loss_div10']:.2g}")
        ax.axvline(sug["lr_at_min_loss"], color="tab:red", ls=":", lw=1.5, label=f"lowest loss at {sug['lr_at_min_loss']:.2g}")
        rows.append(f"{name:14s} suggested peak ~{sug['min_loss_div10']:.3g}   lowest loss at {sug['lr_at_min_loss']:.3g}   steepest {sug['steepest']:.3g}")
    top = max(s[: max(10, len(s) // 3)].max() * 1.3, min(s) * 1.5)
    ax.set_ylim(min(s) * 0.9, top)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate (log scale)")
    ax.set_ylabel("training loss")
    ax.set_title(f"{name}: lr range test")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
os.makedirs(a.out, exist_ok=True)
png = os.path.join(os.path.abspath(a.out), "lr_finder.png")
fig.savefig(png, dpi=150)
print("\n".join(rows))
print("saved:", png)
