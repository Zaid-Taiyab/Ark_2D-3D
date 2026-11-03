"""
lr_finder.py  (Ark_Plus/Pretraining/)  --  learning-rate range test (L. Smith, 2017) for Ark+ pretraining.

Trains a FRESH model on ONE dataset for a few hundred steps while the learning rate rises exponentially
from --finder_min to --finder_max, records the loss of every step, and stops once the loss explodes.
Each step is a real Ark training step: same model and ImageNet init, same optimizer (create_optimizer),
same loss (BCE, or CE for multi-class), same batch size (yaml batch_size for CT, --batch_size for 2D),
same bf16/fp32 setting. At the start of Ark training the teacher-consistency weight is 0, so the loss is
the classification loss only - exactly what this measures.

Run from a folder holding datasets_config.yaml (and the dataset/ symlink for 2D lists), with the same
flags as main_ark.py plus:  --finder_min 1e-5 --finder_max 1 --finder_iters 200 --finder_out <file.csv>
Then plot with plot_lr_finder.py. How to read it: pick a peak lr on the steep downward part of the
curve, about 10x below the lr where the loss is lowest; never where the curve has started rising.
"""
import csv
import math
import os
import sys
import time

import numpy as np


def smooth(losses, beta=0.9):
    """Bias-corrected exponential moving average (the usual lr-finder smoothing)."""
    out, avg = [], 0.0
    for i, l in enumerate(losses, 1):
        avg = beta * avg + (1 - beta) * l
        out.append(avg / (1 - beta ** i))
    return np.array(out)


def suggest(lrs, losses, beta=0.9, skip_start=10, skip_end=3):
    """-> dict with the lr at minimum smoothed loss, that /10, and the steepest-descent lr before the minimum."""
    lrs, s = np.asarray(lrs, float), smooth(losses, beta)
    if len(s) < skip_start + skip_end + 5:
        return {}
    hi = len(s) - skip_end
    i_min = skip_start + int(np.argmin(s[skip_start:hi]))
    g = np.gradient(s, np.log10(lrs))
    i_steep = skip_start + int(np.argmin(g[skip_start:i_min + 1])) if i_min > skip_start else skip_start
    return {"lr_at_min_loss": float(lrs[i_min]), "min_loss_div10": float(lrs[i_min] / 10),
            "steepest": float(lrs[i_steep]), "min_smoothed_loss": float(s[i_min])}


def main():
    import argparse
    fp = argparse.ArgumentParser(add_help=False)
    fp.add_argument("--finder_min", type=float, default=1e-5)
    fp.add_argument("--finder_max", type=float, default=1.0)
    fp.add_argument("--finder_iters", type=int, default=200)
    fp.add_argument("--finder_out", default="")
    fp.add_argument("--finder_config", default="datasets_config.yaml")
    fargs, rest = fp.parse_known_args()
    sys.argv = [sys.argv[0]] + rest                        # the rest are ordinary main_ark.py flags

    import torch
    from torch.utils.data import DataLoader
    from timm.optim import create_optimizer
    from main_ark import get_args_parser                   # same flags and defaults as a real run
    from dataloader import dict_dataloarder
    import dataloader_3d  # noqa: F401  (registers CTVolume)
    from models import build_omni_model
    from utils import get_config

    args = get_args_parser()
    assert args.dataset_list and len(args.dataset_list) == 1, "give exactly one --data_set"
    name = args.dataset_list[0]
    cfg = get_config(fargs.finder_config)[name]
    loader_cls = dict_dataloarder[cfg.get("loader", name)]
    extra = cfg.get("dataset_kwargs", None) or {}
    ds = loader_cls(images_path=cfg["data_dir"], file_path=cfg["train_list"], crop_size=args.crop_size,
                    resize=args.resize, augment=None, **extra)
    bs = int(cfg.get("batch_size", args.batch_size))
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=args.workers, pin_memory=True)
    multiclass = cfg["task_type"] == "multi-class classification"
    criterion = torch.nn.CrossEntropyLoss() if multiclass else torch.nn.BCEWithLogitsLoss()

    model = build_omni_model(args, [len(cfg["diseases"])]).cuda().train()
    optimizer = create_optimizer(args, model)
    n = fargs.finder_iters
    lrs = [fargs.finder_min * (fargs.finder_max / fargs.finder_min) ** (i / (n - 1)) for i in range(n)]
    print(f"[finder] {name}: {len(ds)} training samples, batch {bs}, {n} steps, lr {fargs.finder_min:g} -> "
          f"{fargs.finder_max:g}, opt {args.opt}, momentum {args.momentum}, weight decay {args.weight_decay}, "
          f"amp {bool(args.amp)}, model {args.model_name}", flush=True)

    out = fargs.finder_out or f"lr_finder_{name}.csv"
    rec_lr, rec_loss, best, it, t0 = [], [], float("inf"), iter(dl), time.time()
    for step, lr in enumerate(lrs):
        for g in optimizer.param_groups:
            g["lr"] = lr
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        x, y = batch[0].float().cuda(non_blocking=True), batch[2].float().cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(args.amp)):
            _, pred = model(x, 0)
            loss = criterion(pred.float(), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        l = float(loss.item())
        rec_lr.append(lr)
        rec_loss.append(l)
        s = smooth(rec_loss)[-1]
        best = min(best, s)
        if step % 10 == 0:
            print(f"[finder] step {step:4d}  lr {lr:.3e}  loss {l:.4f}  smoothed {s:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if not math.isfinite(l) or (step > 10 and s > 4 * best):
            print(f"[finder] stopping at step {step}: loss {'not finite' if not math.isfinite(l) else 'exploded'} at lr {lr:.3e}")
            break
    sm = smooth(rec_loss)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "lr", "loss", "smoothed"])
        for i, (a, b, c) in enumerate(zip(rec_lr, rec_loss, sm)):
            w.writerow([i, f"{a:.6e}", f"{b:.6f}", f"{c:.6f}"])
    sug = suggest(rec_lr, [v if math.isfinite(v) else 1e9 for v in rec_loss])
    print(f"[finder] wrote {os.path.abspath(out)}")
    if sug:
        print(f"[finder] {name}: lowest smoothed loss at lr {sug['lr_at_min_loss']:.3g}  ->  suggested peak "
              f"~{sug['min_loss_div10']:.3g}  (steepest descent at {sug['steepest']:.3g})")


if __name__ == "__main__":
    main()
