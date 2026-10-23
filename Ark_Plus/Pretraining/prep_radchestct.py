"""
prep_radchestct.py  (place in Ark_Plus/Pretraining/; needs numpy, scipy, cv2, pandas; no GPU)

One-time RAD-ChestCT (public 3,630-scan release) preprocessing for Ark+ unified 2D/3D training.
Same outputs as prep_ctrate.py, so the rest of the pipeline is unchanged:
  <out>/volumes/<id>.npy   uint8, HU window [-1000, 400], fixed 360 mm field of view (default 160x256x256)
  <out>/drr/<id>.png       512x512 DRR computed from full HU
  <out>/lists/{train,val,test}.txt   from the official imgtrain/imgvalid/imgtest label files
Labels: the 84 abnormality x 52 location matrix is collapsed to whole-volume abnormality labels
(positive if positive in any location); abnormalities with fewer than --min_pos training positives
are dropped (or pass an explicit --keep list).

Step 1 - inspect (prints the file format it found + writes orientation previews):
  python prep_radchestct.py --root /data/jliang12/jpang12/dataset/RAD-ChestCT --out /scratch/ztaiyab/data/RADChestCT_ark --inspect 4
Step 2 - full run (CPU, resumable):
  python prep_radchestct.py --root ... --out ... --workers 32
"""
import argparse
import glob
import os
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
SPLIT_KEYS = (("train", "train"), ("valid", "val"), ("test", "test"))


# --------------------------------------------------------------------------- discovery
def index_npz(root):
    out = {}
    for dirpath, _, files in os.walk(root, onerror=lambda e: print(f"[warn] cannot read {e.filename}")):
        for f in files:
            if f.endswith(".npz"):
                out[f[:-4]] = os.path.join(dirpath, f)
    return out


def load_labels(root, stems, min_pos, keep):
    files = sorted(glob.glob(os.path.join(root, "**", "*Labels*.csv"), recursive=True))
    files = [f for f in files if "abnormal" in os.path.basename(f).lower()] or files
    if not files:
        raise FileNotFoundError(f"no *Labels*.csv under {root}")
    per_split = {}
    for f in files:
        base = os.path.basename(f).lower()
        split = next((s for k, s in SPLIT_KEYS if k in base), None)
        if split is None:
            print(f"[labels] skipping {f} (no train/valid/test in name)")
            continue
        df = pd.read_csv(f)
        # id column = the column whose values best match the .npz file names
        best, best_hits = None, -1
        for c in df.columns[:3]:
            vals = df[c].astype(str).str.replace(".npz", "", regex=False)
            hits = vals.isin(stems).sum()
            if hits > best_hits:
                best, best_hits = c, hits
        ids = df[best].astype(str).str.replace(".npz", "", regex=False)
        lab_cols = [c for c in df.columns if c != best and not str(c).startswith("Unnamed")]
        groups = {}
        for c in lab_cols:
            groups.setdefault(str(c).split("*")[0].strip(), []).append(c)
        vol = pd.DataFrame({ab: (df[cols].apply(pd.to_numeric, errors="coerce").fillna(0) > 0).any(axis=1).astype(int)
                            for ab, cols in groups.items()})
        vol.index = ids
        per_split[split] = vol
        print(f"[labels] {split:5s} {f}: {len(df)} rows, id column '{best}' ({best_hits} match .npz files), "
              f"{len(lab_cols)} label columns -> {len(groups)} abnormalities")
    if "train" not in per_split:
        raise FileNotFoundError("no training label file found")
    pos = per_split["train"].sum(axis=0)
    if keep:
        diseases = [d.strip() for d in keep.split(",")]
        missing = [d for d in diseases if d not in pos.index]
        if missing:
            raise KeyError(f"--keep names not found: {missing}. Available: {sorted(pos.index)}")
    else:
        diseases = [d for d in pos.index if pos[d] >= min_pos]
    print(f"[labels] keeping {len(diseases)} abnormalities with >= {min_pos} train positives "
          f"(dropped {len(pos) - len(diseases)} rarer ones)")
    print("[labels] train positives:", ", ".join(f"{d}={int(pos[d])}" for d in diseases))
    labels, split_of = {}, {}
    for split, vol in per_split.items():
        for vid, row in vol[diseases].iterrows():
            labels[vid] = [int(v) for v in row.values]
            split_of[vid] = split
    return labels, split_of, diseases


# --------------------------------------------------------------------------- per volume
def load_hu_zyx(path, cfg):
    z = np.load(path)
    key = cfg.key if cfg.key in z.files else z.files[0]
    arr = np.asarray(z[key], dtype=np.float32)
    info = {"npz_keys": z.files, "key_used": key, "stored_shape": arr.shape}
    arr = arr.transpose([int(a) for a in cfg.axes.split(",")])
    for ax, name in enumerate("zyx"):
        if name in cfg.flips:
            arr = np.flip(arr, axis=ax)
    arr = np.ascontiguousarray(arr)
    p1 = float(np.percentile(arr[::4, ::4, ::4], 1))
    if cfg.hu_offset == "auto":
        offset = 0.0 if p1 < -500 else -1024.0
    else:
        offset = float(cfg.hu_offset)
    hu = arr + offset
    info["hu"] = f"stored p1={p1:.0f} -> offset {offset:+.0f}" + ("" if offset == 0 else "  (CHECK: values were not HU)")
    info["hu_p1_p99"] = tuple(round(float(v)) for v in np.percentile(hu[::4, ::4, ::4], (1, 99)))
    spacing = tuple(float(s) for s in cfg.spacing.split(","))
    info["zyx_shape"], info["spacing"], info["flips"] = hu.shape, spacing, cfg.flips or "none"
    return hu, spacing, info


def init(cfg):
    global CFG
    CFG = cfg
    cv2.setNumThreads(0)


def work(job):
    vid, path = job
    vol_p = os.path.join(CFG.out, "volumes", vid + ".npy")
    drr_p = os.path.join(CFG.out, "drr", vid + ".png")
    if not CFG.inspect and os.path.exists(vol_p) and os.path.exists(drr_p):
        return vid, True, "exists"
    try:
        hu, spacing, info = load_hu_zyx(path, CFG)
        fixed, drr, target = process_hu(hu, spacing, CFG)
        if CFG.inspect:
            cv2.imwrite(os.path.join(CFG.out, "inspect", vid + ".png"), preview_panel(fixed, drr, target))
            return vid, True, info
        np.save(vol_p, window_uint8(fixed))
        cv2.imwrite(drr_p, drr)
        return vid, True, "ok"
    except Exception:
        return vid, False, traceback.format_exc(limit=3)


def write_lists(out, labels, split_of, diseases):
    out = os.path.abspath(out)
    counts = {}
    for s in ("train", "val", "test"):
        rows = sorted(v for v in labels if split_of[v] == s and os.path.exists(os.path.join(out, "volumes", v + ".npy")))
        with open(os.path.join(out, "lists", f"{s}.txt"), "w") as f:
            for v in rows:
                f.write(f"volumes/{v}.npy " + " ".join(map(str, labels[v])) + "\n")
        counts[s] = len(rows)
    print(f"[lists] {counts} -> {os.path.join(out, 'lists')}")
    d = ", ".join(f"'{x}'" for x in diseases)
    print(f"""
# ---- add to datasets_config.yaml ----
RADChestCT:
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
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--inspect", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--key", default="ct", help="array name inside the .npz (falls back to the first array)")
    p.add_argument("--axes", default="0,1,2", help="transpose of the stored array to (z,y,x)")
    p.add_argument("--flips", default="yx", help="axes to flip after transposing, any of 'zyx' ('' = none). "
                                                 "'yx' is right for arrays built from DICOM pixel data")
    p.add_argument("--spacing", default="0.8,0.8,0.8", help="z,y,x voxel size in mm of the stored arrays")
    p.add_argument("--hu_offset", default="auto", help="'auto', or a number added to stored values to get HU")
    p.add_argument("--fov", default="360,360,360")
    p.add_argument("--shape", default="160,256,256")
    p.add_argument("--drr_px", type=int, default=512)
    p.add_argument("--min_pos", type=int, default=20, help="drop abnormalities with fewer train positives")
    p.add_argument("--keep", default="", help="explicit comma-separated abnormality list (overrides --min_pos)")
    p.add_argument("--lists_only", action="store_true")
    cfg = p.parse_args()
    cfg.fov = tuple(float(v) for v in cfg.fov.split(","))
    cfg.shape = tuple(int(v) for v in cfg.shape.split(","))
    cfg.flips = cfg.flips.lower()

    for sub in ("volumes", "drr", "inspect", "lists"):
        os.makedirs(os.path.join(cfg.out, sub), exist_ok=True)
    stems = index_npz(cfg.root)
    print(f"[files] {len(stems)} .npz volumes under {cfg.root}")
    labels, split_of, diseases = load_labels(cfg.root, set(stems), cfg.min_pos, cfg.keep)
    vids = sorted(v for v in labels if v in stems)
    print(f"[files] {len(vids)} volumes have labels; {len(labels) - len(vids)} labelled ids without a .npz")
    if cfg.lists_only:
        write_lists(cfg.out, labels, split_of, diseases)
        return
    if cfg.inspect:
        vids = [v for v in vids if split_of[v] == "train"][:cfg.inspect]
    jobs = [(v, stems[v]) for v in vids]
    n_ok, failed = 0, []
    with Pool(max(1, min(cfg.workers, len(jobs))), initializer=init, initargs=(cfg,)) as pool:
        for i, (vid, ok, info) in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            if ok:
                n_ok += 1
                if cfg.inspect:
                    print(f"\n{vid}:")
                    for k, v in info.items():
                        print(f"   {k:13s} {v}")
            else:
                failed.append(vid)
                print(f"[FAIL] {vid}\n{info}")
            if not cfg.inspect and (i + 1) % 200 == 0:
                print(f"[progress] {i + 1}/{len(jobs)} ok={n_ok} failed={len(failed)}", flush=True)
    if cfg.inspect:
        print(f"\nPreviews in {os.path.join(cfg.out, 'inspect')}. Check: axial = sternum at top, heart on the image "
              "right; coronal = head at top; DRR = frontal CXR look, heart on the image right. Fix with --flips "
              "(e.g. 'zyx', 'x', '') or --axes until all three are right.")
        return
    with open(os.path.join(cfg.out, "failed.txt"), "w") as f:
        f.write("\n".join(failed))
    print(f"[done] ok={n_ok} failed={len(failed)}")
    write_lists(cfg.out, labels, split_of, diseases)


if __name__ == "__main__":
    main()
