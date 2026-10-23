# Plot VinDr: original code vs LR fix, student + teacher mAUC, and the learning rate each run used.
import os, re, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("ARK_RUNS", "/scratch/ztaiyab/ark_runs")
RUNS = [
    ("Original code", "arkorig_swinb224_imnet22k_vindr_ep200", "tab:blue"),
    ("LR fix", "arkorig-lrfix_swinb224_imnet22k_vindr_ep200", "tab:orange"),
]
OUT = os.path.join(ROOT, "plots")
os.makedirs(OUT, exist_ok=True)

def read_auc(run):
    ep, rows = None, {}
    for line in open(os.path.join(ROOT, run, "test_results.txt")):
        m = re.match(r"Epoch (\d+):", line)
        if m:
            ep = int(m.group(1))
        m = re.search(r"VinDrCXR: Student mAUC = ([\d.]+), Teacher mAUC = ([\d.]+)", line)
        if m and ep is not None:
            rows[ep] = (float(m.group(1)), float(m.group(2)))
    return sorted((e, s, t) for e, (s, t) in rows.items())

def read_valloss(run):
    out = {}
    for line in open(os.path.join(ROOT, run, "train_log.txt")):
        m = re.match(r"Epoch (\d+): avg_val_loss = ([\d.]+)", line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out

def lr_fixed(t, peak=0.3, w0=1e-6, warm=20, lmin=1e-5, T=200):
    if t < warm:
        return w0 + t * (peak - w0) / warm
    if t < T:
        return lmin + 0.5 * (peak - lmin) * (1 + math.cos(math.pi * t / T))
    return lmin

def lr_original(vl, peak=0.3, w0=1e-6, warm=20):
    # epoch 0 trains at w0; epoch e >= 1 trains at the warm-up formula evaluated at the previous epoch's val loss
    return {e: (w0 if e == 0 else w0 + vl[e - 1] * (peak - w0) / warm) for e in range(0, 200) if e == 0 or (e - 1) in vl}

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
summary = []
for name, run, color in RUNS:
    rows = [r for r in read_auc(run) if r[0] > 0]
    ep = [r[0] for r in rows]
    ax.plot(ep, [r[2] for r in rows], "-o", color=color, lw=2.2, ms=4, label=f"{name}: teacher")
    ax.plot(ep, [r[1] for r in rows], "--", color=color, lw=1.4, alpha=0.75, label=f"{name}: student")
    last = rows[-1]
    best_t = max(rows, key=lambda r: r[2])
    summary.append(f"{name:14s} final teacher {last[2]:.4f}  student {last[1]:.4f}  | best teacher {best_t[2]:.4f} @ epoch {best_t[0]}")
    if run.startswith("arkorig-lrfix"):
        xs = list(range(0, 200))
        ax2.plot(xs, [lr_fixed(e) for e in xs], color=color, lw=2, label=f"{name} (warm-up to 0.3, cosine)")
    else:
        lr = lr_original(read_valloss(run))
        xs = sorted(lr)
        ax2.plot(xs, [lr[e] for e in xs], color=color, lw=2, label=f"{name} (from logged val loss)")

ax.axhline(0.9035, color="grey", ls=":", lw=1.5)
ax.text(2, 0.9045, "Ark paper: ImageNet-supervised Swin-B, 90.35", color="grey", fontsize=9, va="bottom")
ax.set_ylabel("VinDr-CXR test mAUC")
ax.set_ylim(0.68, 0.93)
ax.set_title("VinDr-CXR, Swin-B 224, ImageNet init, 200 epochs: original code vs LR fix\n(epoch 0, untrained heads at ~0.5 AUC, not shown)", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
ax2.set_yscale("log")
ax2.set_ylabel("learning rate")
ax2.set_xlabel("epoch")
ax2.grid(alpha=0.3, which="both")
ax2.legend(loc="lower left", fontsize=9)
fig.tight_layout()
png = os.path.join(OUT, "vindr_orig_vs_lrfix.png")
pdf = os.path.join(OUT, "vindr_orig_vs_lrfix.pdf")
fig.savefig(png, dpi=160)
fig.savefig(pdf)
print("\n".join(summary))
print("saved:", png)
print("saved:", pdf)
print("folder:", OUT)
