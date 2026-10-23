"""
prep_ctrate.py  (place in Ark_Plus/Pretraining/; needs numpy, scipy, cv2, nibabel, pandas; no GPU)

One-time CT-RATE preprocessing for Ark+ unified 2D/3D training. Per volume it
  1. reads the array and converts to HU (header scaling if present, else RescaleSlope/Intercept from
     the CT-RATE metadata CSV; detected per file, so both the original and the header-fixed releases work)
  2. orients to (z,y,x) RAS: from the header if it is trustworthy, else raw array + metadata spacing
     + your --raw_axes/--flip_* settings (verify with --inspect)
  3. resamples to a fixed physical field of view (default 360 mm cube -> 160x256x256)
  4. saves  volumes/<name>.npy  (uint8, HU window [-1000, 400])  and  drr/<name>.png  (512x512 DRR from full HU)
and writes patient-level train/val/test lists plus the datasets_config.yaml block.

Step 1 - inspect a few volumes, look at <out>/inspect/*.png, adjust flags if needed:
  python prep_ctrate.py --root /data/jliang12/jpang12/dataset/CT-RATE --out /scratch/ztaiyab/data/CTRATE_ark --inspect 4
Step 2 - full run (CPU job, resumable):
  python prep_ctrate.py --root ... --out ... --workers 32
"""
import argparse
import hashlib
import os
import re
import sys
import traceback
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ct_utils import resample, center_fit, window_uint8, make_drr, preview_panel  # noqa: E402

NAME_RE = re.compile(r"^(train|valid)_(\d+)_([a-z]+)_(\d+)$")
CFG = None


# --------------------------------------------------------------------------- discovery
def find_one(root, filename):
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return os.path.join(dirpath, filename)
    return None


def index_niftis(root):
    out = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".nii.gz") and NAME_RE.match(f[:-7]):
                out[f[:-7]] = os.path.join(dirpath, f)
    return out


def parse_xy(v):
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(v))
    return float(nums[0]), float(nums[1] if len(nums) > 1 else nums[0])


def col(df, *names):
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"none of {names} in columns {list(df.columns)}")


def load_tables(root):
    meta, labels = [], []
    for split, mfile, lfile in (("train", "train_metadata.csv", "train_predicted_labels.csv"),
                                ("valid", "validation_metadata.csv", "valid_predicted_labels.csv")):
        mp, lp = find_one(root, mfile), find_one(root, lfile)
        if mp is None or lp is None:
            raise FileNotFoundError(f"could not find {mfile} / {lfile} under {root}")
        m, lab = pd.read_csv(mp), pd.read_csv(lp)
        print(f"[tables] {split}: metadata {mp} ({len(m)} rows), labels {lp} ({len(lab)} rows)")
        meta.append(m)
        labels.append(lab)
    meta, labels = pd.concat(meta, ignore_index=True), pd.concat(labels, ignore_index=True)
    vn_m, vn_l = col(meta, "VolumeName"), col(labels, "VolumeName")
    meta["name"] = meta[vn_m].astype(str).str.replace(".nii.gz", "", regex=False)
    labels["name"] = labels[vn_l].astype(str).str.replace(".nii.gz", "", regex=False)
    diseases = [c for c in labels.columns if c not in (vn_l, "name")]
    c_slope, c_inter = col(meta, "RescaleSlope"), col(meta, "RescaleIntercept")
    c_xy, c_z = col(meta, "XYSpacing", "XY_Spacing"), col(meta, "ZSpacing", "Z_Spacing")
    md = {}
    for r in meta.to_dict("records"):
        sx, sy = parse_xy(r[c_xy])
        md[r["name"]] = dict(slope=float(r[c_slope]), inter=float(r[c_inter]), sx=sx, sy=sy, sz=float(r[c_z]))
    lab = {r["name"]: [int(round(float(r[d]))) for d in diseases] for r in labels.to_dict("records")}
    return md, lab, diseases


# --------------------------------------------------------------------------- per volume
def load_hu_zyx(path, m, cfg):
    """Returns HU volume (z,y,x), spacing (sz,sy,sx) and a diagnostics dict."""
    import nibabel as nib
    img = nib.load(path)
    zooms = [float(z) for z in img.header.get_zooms()[:3]]
    info = {"raw_shape": tuple(img.shape), "hdr_zooms": tuple(round(z, 3) for z in zooms),
            "meta_spacing": (m["sz"], m["sy"], m["sx"])}
    orient = cfg.orient
    if orient == "auto":
        aff_ok = not np.allclose(img.affine[:3, :3], np.eye(3))
        sp_ok = np.allclose(sorted(zooms), sorted([m["sx"], m["sy"], m["sz"]]), rtol=0.05)
        orient = "header" if (aff_ok and sp_ok) else "raw"
    info["orient"] = orient

    if orient == "header":
        can = nib.as_closest_canonical(img)
        arr = np.asarray(can.dataobj, dtype=np.float32).transpose(2, 1, 0)
        zx, zy, zz = [float(z) for z in can.header.get_zooms()[:3]]
        spacing = (zz, zy, zx)
        proxy = can.dataobj
    else:
        axes = [int(a) for a in cfg.raw_axes.split(",")]
        arr = np.asarray(img.dataobj, dtype=np.float32).transpose(axes)
        for ax, flip in enumerate((cfg.flip_z, cfg.flip_y, cfg.flip_x)):
            if flip:
                arr = np.flip(arr, axis=ax)
        arr = np.ascontiguousarray(arr)
        spacing = (m["sz"], m["sy"], m["sx"])
        proxy = img.dataobj

    slope, inter = getattr(proxy, "slope", 1.0), getattr(proxy, "inter", 0.0)
    hdr_scaled = not (np.isclose(slope, 1.0) and np.isclose(inter, 0.0))
    p1 = float(np.percentile(arr[::4, ::4, ::4], 1))
    if hdr_scaled or p1 < -500:
        hu, info["hu"] = arr, f"as stored (header scaling={hdr_scaled}, p1={p1:.0f})"
    else:
        hu = arr * m["slope"] + m["inter"]
        info["hu"] = f"metadata slope {m['slope']} / intercept {m['inter']} (raw p1={p1:.0f})"
    info["hu_p1_p99"] = tuple(round(float(v)) for v in np.percentile(hu[::4, ::4, ::4], (1, 99)))
    return hu, spacing, info


def process_hu(hu, spacing, cfg):
    target = tuple(f / s for f, s in zip(cfg.fov, cfg.shape))
    fixed = center_fit(resample(hu, spacing, target), cfg.shape, fill=-1024.0)
    drr = make_drr(fixed, size=cfg.drr_px, spacing_zx=(target[0], target[2]))
    return fixed, drr, target


def init(cfg):
    global CFG
    CFG = cfg
    cv2.setNumThreads(0)


def work(job):
    name, path, m = job
    vol_p = os.path.join(CFG.out, "volumes", name + ".npy")
    drr_p = os.path.join(CFG.out, "drr", name + ".png")
    if not CFG.inspect and os.path.exists(vol_p) and os.path.exists(drr_p):
        return name, True, "exists"
    try:
        hu, spacing, info = load_hu_zyx(path, m, CFG)
        fixed, drr, target = process_hu(hu, spacing, CFG)
        if CFG.inspect:
            cv2.imwrite(os.path.join(CFG.out, "inspect", name + ".png"), preview_panel(fixed, drr, target))
            return name, True, info
        np.save(vol_p, window_uint8(fixed))
        cv2.imwrite(drr_p, drr)
        return name, True, info["orient"]
    except Exception:
        return name, False, traceback.format_exc(limit=3)


# --------------------------------------------------------------------------- splits
def select(names, recons):
    """recons='first': one reconstruction per scan (lowest recon id); 'all': every reconstruction."""
    if recons == "all":
        return sorted(names)
    best = {}
    for n in names:
        split, pid, scan, rec = NAME_RE.match(n).groups()
        key = (split, pid, scan)
        if key not in best or int(rec) < int(NAME_RE.match(best[key]).group(4)):
            best[key] = n
    return sorted(best.values())


def is_val_patient(pid, frac, seed):
    h = int(hashlib.md5(f"{seed}-{pid}".encode()).hexdigest(), 16)
    return (h % 10000) < frac * 10000


def write_lists(out, names, lab, diseases, cfg):
    out = os.path.abspath(out)
    os.makedirs(os.path.join(out, "lists"), exist_ok=True)
    split_of = {}
    for n in names:
        split, pid = NAME_RE.match(n).group(1), NAME_RE.match(n).group(2)
        split_of[n] = "test" if split == "valid" else ("val" if is_val_patient(pid, cfg.val_frac, cfg.seed) else "train")
    counts = {}
    for s in ("train", "val", "test"):
        rows = [n for n in names if split_of[n] == s
                and os.path.exists(os.path.join(out, "volumes", n + ".npy"))]
        with open(os.path.join(out, "lists", f"{s}.txt"), "w") as f:
            for n in rows:
                f.write(f"volumes/{n}.npy " + " ".join(str(v) for v in lab[n]) + "\n")
        counts[s] = len(rows)
    print(f"[lists] {counts} -> {os.path.join(out, 'lists')}")
    d = ", ".join(f"'{x}'" for x in diseases)
    print(f"""
# ---- add to datasets_config.yaml ----
CTRATE:
  loader: CTVolume
  dims: 3
  data_dir: {out}
  train_list: {out}/lists/train.txt
  val_list: {out}/lists/val.txt
  test_list: {out}/lists/test.txt      # = official CT-RATE validation set
  diseases: [{d}]
  task_type: multi-label classification
  batch_size: 8                        # set from check_unified.py (multiple of GPU count)
  accum_steps: 1
  drr_head: true
  dataset_kwargs:
    volume_size: [128, 224, 224]       # unified_swin_base; [128, 192, 192] for window-12 models
    drr_dir: drr
    drr_size: 224                      # = --input_size
    return_drr: true
""")


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="CT-RATE folder (searched recursively)")
    p.add_argument("--out", required=True)
    p.add_argument("--inspect", type=int, default=0, help="only write previews for N training volumes")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--recons", choices=["first", "all"], default="first")
    p.add_argument("--orient", choices=["auto", "header", "raw"], default="auto")
    p.add_argument("--raw_axes", default="2,1,0", help="transpose of the raw array to (z,y,x)")
    p.add_argument("--flip_z", action="store_true")
    p.add_argument("--flip_y", action="store_true")
    p.add_argument("--flip_x", action="store_true")
    p.add_argument("--fov", default="360,360,360", help="z,y,x field of view in mm")
    p.add_argument("--shape", default="160,256,256", help="stored z,y,x size")
    p.add_argument("--drr_px", type=int, default=512)
    p.add_argument("--val_frac", type=float, default=0.05, help="fraction of TRAIN patients held out for val")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lists_only", action="store_true", help="just (re)write lists from existing volumes")
    cfg = p.parse_args()
    cfg.fov = tuple(float(v) for v in cfg.fov.split(","))
    cfg.shape = tuple(int(v) for v in cfg.shape.split(","))

    for sub in ("volumes", "drr", "inspect", "lists"):
        os.makedirs(os.path.join(cfg.out, sub), exist_ok=True)
    md, lab, diseases = load_tables(cfg.root)
    print(f"[tables] {len(diseases)} labels: {diseases}")
    niftis = index_niftis(cfg.root)
    print(f"[files] {len(niftis)} NIfTI volumes found under {cfg.root}")
    names = select([n for n in niftis if n in md and n in lab], cfg.recons)
    print(f"[files] {len(names)} selected (recons={cfg.recons}); "
          f"stored size ~{np.prod(cfg.shape) / 2**20:.1f} MB each, ~{len(names) * np.prod(cfg.shape) / 2**40:.2f} TB total")

    if cfg.lists_only:
        write_lists(cfg.out, names, lab, diseases, cfg)
        return
    if cfg.inspect:
        names = [n for n in names if n.startswith("train_")][:cfg.inspect]
    jobs = [(n, niftis[n], md[n]) for n in names]
    n_ok, failed = 0, []
    with Pool(cfg.workers if not cfg.inspect else min(cfg.workers, len(jobs)), initializer=init, initargs=(cfg,)) as pool:
        for i, (name, ok, info) in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            if ok:
                n_ok += 1
                if cfg.inspect:
                    print(f"\n{name}:")
                    for k, v in info.items():
                        print(f"   {k:13s} {v}")
            else:
                failed.append(name)
                print(f"[FAIL] {name}\n{info}")
            if not cfg.inspect and (i + 1) % 200 == 0:
                print(f"[progress] {i + 1}/{len(jobs)}  ok={n_ok}  failed={len(failed)}", flush=True)
    if cfg.inspect:
        print(f"\nPreviews in {os.path.join(cfg.out, 'inspect')}. Check: axial = anterior (sternum) at top, "
              "patient's right on the image left (heart/aorta arch on the image right); coronal = head at top; "
              "DRR = looks like a frontal CXR, heart on the image right. If not, rerun with --orient raw and "
              "--flip_z/--flip_y/--flip_x (or a different --raw_axes) until it does.")
        return
    with open(os.path.join(cfg.out, "failed.txt"), "w") as f:
        f.write("\n".join(failed))
    print(f"[done] ok={n_ok} failed={len(failed)} (see failed.txt)")
    write_lists(cfg.out, names, lab, diseases, cfg)


if __name__ == "__main__":
    main()
