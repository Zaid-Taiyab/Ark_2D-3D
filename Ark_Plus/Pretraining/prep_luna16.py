"""
prep_luna16.py  (place in Ark_Plus/Pretraining/; needs numpy, scipy, cv2, pandas; no GPU, no SimpleITK)

One-time LUNA16 preprocessing for Ark+ unified 2D/3D training. Same outputs as prep_ctrate.py:
  <out>/volumes/<seriesuid>.npy   uint8, HU window [-1000, 400], fixed 360 mm field of view (160x256x256)
  <out>/drr/<seriesuid>.png       512x512 DRR from full HU
  <out>/lists/{train,val,test}.txt
Scan-level labels from annotations.csv (nodules >= 3 mm accepted by >= 3 of 4 radiologists):
  nodule_any, nodule_ge6mm, nodule_ge10mm   (thresholds configurable with --thresholds)
Splits follow the official 10 subsets (default: subset0-6 train, 7 val, 8-9 test).

Get annotations.csv once (open data, LUNA16 Zenodo record 3723295):
  wget -O /scratch/ztaiyab/data/LUNA16_meta/annotations.csv "https://zenodo.org/records/3723295/files/annotations.csv?download=1"
Step 1 - inspect:
  python prep_luna16.py --root /data/jliang12/jpang12/dataset/LUNA16/org --annotations /scratch/ztaiyab/data/LUNA16_meta/annotations.csv --out /scratch/ztaiyab/data/LUNA16_ark --inspect 4
Step 2 - full run (CPU, resumable): same without --inspect, with --workers 16
"""
import argparse
import os
import re
import sys
import traceback
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ct_utils import window_uint8, preview_panel  # noqa: E402
from prep_ctrate import process_hu  # noqa: E402

CFG = None
MET_DTYPES = {"MET_SHORT": "i2", "MET_USHORT": "u2", "MET_FLOAT": "f4", "MET_DOUBLE": "f8",
              "MET_UCHAR": "u1", "MET_CHAR": "i1", "MET_INT": "i4", "MET_UINT": "u4"}


# --------------------------------------------------------------------------- MetaImage reader
def read_mhd(path):
    hdr = {}
    with open(path, "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=", 1)
                hdr[k.strip()] = v.strip()
    dims = [int(v) for v in hdr["DimSize"].split()]                      # x y z
    spacing = [float(v) for v in hdr.get("ElementSpacing", hdr.get("ElementSize", "1 1 1")).split()]
    tm = [float(v) for v in hdr.get("TransformMatrix", "1 0 0 0 1 0 0 0 1").split()]
    order = ">" if hdr.get("BinaryDataByteOrderMSB", hdr.get("ElementByteOrderMSB", "False")).lower() == "true" else "<"
    dtype = np.dtype(order + MET_DTYPES[hdr["ElementType"]])
    raw = os.path.join(os.path.dirname(path), hdr["ElementDataFile"])
    arr = np.fromfile(raw, dtype=dtype, offset=int(hdr.get("HeaderSize", "0")))
    arr = arr.reshape(dims[2], dims[1], dims[0])                         # (z, y, x) index order
    return arr, spacing, np.array(tm).reshape(3, 3), hdr


def to_ras_zyx(arr, tm):
    """MetaImage world coordinates are LPS. Flip index axes so that x -> patient Right,
    y -> Anterior, z -> Superior (our RAS (z,y,x) convention). Axis-aligned scans only."""
    off_diag = np.abs(tm - np.diag(np.diag(tm))).max()
    if off_diag > 1e-3:
        raise ValueError(f"non axis-aligned TransformMatrix {tm.ravel().tolist()}")
    sx, sy, sz = np.sign(np.diag(tm))
    flips = []
    if sx > 0:      # index x increases toward patient Left (LPS +x) -> flip
        arr = arr[:, :, ::-1]
        flips.append("x")
    if sy > 0:      # index y increases toward Posterior (LPS +y) -> flip
        arr = arr[:, ::-1, :]
        flips.append("y")
    if sz < 0:      # index z increases toward Inferior -> flip
        arr = arr[::-1, :, :]
        flips.append("z")
    return arr, "".join(flips) or "none"


# --------------------------------------------------------------------------- labels / splits
def load_labels(ann_path, uids, thresholds):
    ann = pd.read_csv(ann_path)
    for c in ("seriesuid", "diameter_mm"):
        if c not in ann.columns:
            raise KeyError(f"{ann_path} has no column '{c}' (columns: {list(ann.columns)})")
    max_d = ann.groupby("seriesuid")["diameter_mm"].max()
    names = ["nodule_any" if t <= 0 else f"nodule_ge{t:g}mm" for t in thresholds]
    labels = {}
    for u in uids:
        d = float(max_d.get(u, 0.0))
        labels[u] = [int(u in max_d.index and d >= t) for t in thresholds]
    pos = np.array(list(labels.values())).sum(0)
    print(f"[labels] {len(ann)} nodules in {ann['seriesuid'].nunique()} scans; "
          f"{len(set(ann['seriesuid']) - set(uids))} annotated uids have no .mhd here")
    print("[labels] positives over all scans:", ", ".join(f"{n}={int(p)}" for n, p in zip(names, pos)))
    return labels, names


def parse_splits(spec):
    out = {}
    for part in spec.split(","):
        rng, split = part.split(":")
        a, b = (rng.split("-") + [rng])[:2]
        for i in range(int(a), int(b) + 1):
            out[i] = split
    return out


# --------------------------------------------------------------------------- per volume
def load_hu_zyx(path, cfg):
    arr, spacing_xyz, tm, hdr = read_mhd(path)
    arr, flips = to_ras_zyx(arr.astype(np.float32), tm)
    for ax, name in enumerate("zyx"):
        if name in cfg.extra_flips:
            arr = np.flip(arr, axis=ax)
    arr = np.ascontiguousarray(arr)
    p1 = float(np.percentile(arr[::4, ::4, ::4], 1))
    offset = 0.0 if p1 < -500 else -1024.0
    hu = arr + offset
    spacing = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    info = {"dims_xyz": hdr["DimSize"], "spacing_zyx": tuple(round(s, 3) for s in spacing),
            "transform": hdr.get("TransformMatrix", "(none)"), "flips": f"{flips} + extra '{cfg.extra_flips}'",
            "hu": f"stored p1={p1:.0f} -> offset {offset:+.0f}" + ("" if offset == 0 else "  (CHECK: not HU?)"),
            "hu_p1_p99": tuple(round(float(v)) for v in np.percentile(hu[::4, ::4, ::4], (1, 99)))}
    return hu, spacing, info


def init(cfg):
    global CFG
    CFG = cfg
    cv2.setNumThreads(0)


def work(job):
    uid, path = job
    vol_p = os.path.join(CFG.out, "volumes", uid + ".npy")
    drr_p = os.path.join(CFG.out, "drr", uid + ".png")
    if not CFG.inspect and os.path.exists(vol_p) and os.path.exists(drr_p):
        return uid, True, "exists"
    try:
        hu, spacing, info = load_hu_zyx(path, CFG)
        fixed, drr, target = process_hu(hu, spacing, CFG)
        if CFG.inspect:
            cv2.imwrite(os.path.join(CFG.out, "inspect", uid[-12:] + ".png"), preview_panel(fixed, drr, target))
            return uid, True, info
        np.save(vol_p, window_uint8(fixed))
        cv2.imwrite(drr_p, drr)
        return uid, True, "ok"
    except Exception:
        return uid, False, traceback.format_exc(limit=3)


def write_lists(out, uids, subset_of, split_map, labels, names):
    out = os.path.abspath(out)
    counts = {}
    for s in ("train", "val", "test"):
        rows = sorted(u for u in uids if split_map.get(subset_of[u]) == s
                      and os.path.exists(os.path.join(out, "volumes", u + ".npy")))
        with open(os.path.join(out, "lists", f"{s}.txt"), "w") as f:
            for u in rows:
                f.write(f"volumes/{u}.npy " + " ".join(map(str, labels[u])) + "\n")
        pos = np.array([labels[u] for u in rows]).sum(0) if rows else [0] * len(names)
        counts[s] = f"{len(rows)} scans, positives " + "/".join(str(int(p)) for p in pos)
    print(f"[lists] {counts} -> {os.path.join(out, 'lists')}")
    d = ", ".join(f"'{x}'" for x in names)
    print(f"""
# ---- add to datasets_config.yaml ----
LUNA16:
  loader: CTVolume
  dims: 3
  data_dir: {out}
  train_list: {out}/lists/train.txt
  val_list: {out}/lists/val.txt
  test_list: {out}/lists/test.txt
  diseases: [{d}]
  task_type: multi-label classification
  batch_size: 8                        # set from check_unified.py (multiple of GPU count)
  accum_steps: 1
  drr_head: true
  dataset_kwargs:
    volume_size: [128, 224, 224]       # unified_swin_base
    drr_dir: drr
    drr_size: 224                      # = --input_size
    return_drr: true
""")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="folder containing subset0 ... subset9")
    p.add_argument("--annotations", required=True, help="LUNA16 annotations.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--inspect", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--thresholds", default="0,6,10", help="nodule diameter thresholds (mm) -> one label each")
    p.add_argument("--splits", default="0-6:train,7:val,8-9:test", help="subset index -> split")
    p.add_argument("--extra_flips", default="", help="additional flips if the previews look wrong, any of 'zyx'")
    p.add_argument("--fov", default="360,360,360")
    p.add_argument("--shape", default="160,256,256")
    p.add_argument("--drr_px", type=int, default=512)
    p.add_argument("--lists_only", action="store_true")
    cfg = p.parse_args()
    cfg.fov = tuple(float(v) for v in cfg.fov.split(","))
    cfg.shape = tuple(int(v) for v in cfg.shape.split(","))
    cfg.extra_flips = cfg.extra_flips.lower()

    for sub in ("volumes", "drr", "inspect", "lists"):
        os.makedirs(os.path.join(cfg.out, sub), exist_ok=True)
    mhd, subset_of = {}, {}
    for dirpath, _, files in os.walk(cfg.root, onerror=lambda e: print(f"[warn] cannot read {e.filename}")):
        m = re.search(r"subset(\d+)", dirpath)
        for f in files:
            if f.endswith(".mhd"):
                uid = f[:-4]
                mhd[uid] = os.path.join(dirpath, f)
                subset_of[uid] = int(m.group(1)) if m else -1
    print(f"[files] {len(mhd)} .mhd scans under {cfg.root}")
    uids = sorted(mhd)
    labels, names = load_labels(cfg.annotations, uids, [float(t) for t in cfg.thresholds.split(",")])
    split_map = parse_splits(cfg.splits)

    if cfg.lists_only:
        write_lists(cfg.out, uids, subset_of, split_map, labels, names)
        return
    if cfg.inspect:
        uids = uids[:cfg.inspect]
    jobs = [(u, mhd[u]) for u in uids]
    n_ok, failed = 0, []
    with Pool(max(1, min(cfg.workers, len(jobs))), initializer=init, initargs=(cfg,)) as pool:
        for i, (uid, ok, info) in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            if ok:
                n_ok += 1
                if cfg.inspect:
                    print(f"\n...{uid[-12:]}  (subset{subset_of[uid]}, labels {labels[uid]}):")
                    for k, v in info.items():
                        print(f"   {k:12s} {v}")
            else:
                failed.append(uid)
                print(f"[FAIL] {uid}\n{info}")
            if not cfg.inspect and (i + 1) % 100 == 0:
                print(f"[progress] {i + 1}/{len(jobs)} ok={n_ok} failed={len(failed)}", flush=True)
    if cfg.inspect:
        print(f"\nPreviews in {os.path.join(cfg.out, 'inspect')}. Check: axial = sternum at top, heart on the image "
              "right; coronal = head at top; DRR = frontal CXR look, heart on the image right. "
              "If not, rerun with --extra_flips (e.g. 'x', 'z').")
        return
    with open(os.path.join(cfg.out, "failed.txt"), "w") as f:
        f.write("\n".join(failed))
    print(f"[done] ok={n_ok} failed={len(failed)}")
    write_lists(cfg.out, sorted(mhd), subset_of, split_map, labels, names)


if __name__ == "__main__":
    main()
