"""
prep_ricord.py  (Ark_Plus/Pretraining/; runs inside ark_vindr.sif: needs only numpy, scipy, cv2, pydicom)

MIDRC-RICORD chest CT -> the Ark CT format (same outputs as prep_ctrate.py / prep_luna16.py):
  1a (COVID-19 positive)  -> label 1        1b (COVID-19 negative) -> label 0
  <out>/volumes/<id>.npy   uint8, HU window [-1000, 400], fixed 360 mm cube (160x256x256)
  <out>/drr/<id>.png       512x512 DRR from full HU
  <out>/lists/{train,val,test}.txt          patient-level split, stratified by label (default 70/10/20)
  <out>/series_index.csv   every CT series found, and which one was used for each study
  <out>/balance_report.txt 1a vs 1b: scanners, kernels, slice thickness, contrast -> check for shortcuts
One series per study: axial CT, not a localizer, >= 40 slices; prefers a standard (soft) kernel, then the
slice thickness closest to --prefer_mm (3 mm), then the most slices, so 1a and 1b get comparable series.
Splits: --unimiss_lists <dir> uses the UniMiSS+ RICORD_{train,val,test}.txt (patient-level). With the default
--samples listed, every listed series is one sample (their protocol, incl. coronal/sagittal reformats);
--samples per_study keeps one series per study with the same patient split. Without lists: patient-level
stratified split (--split) or your own ID lists (--split_dir).

Step 1 - inspect (2 studies per release, previews + series info):
  python prep_ricord.py --root_1a /scratch/ztaiyab/data/RICORD/1a_CT_covid_pos --root_1b /scratch/ztaiyab/data/RICORD/1b_CT_covid_neg --out /scratch/ztaiyab/data/RICORD_ark --inspect 4
Step 2 - full run (CPU, resumable): same without --inspect, with --workers 16
"""
import argparse
import hashlib
import os
import re
import sys
import traceback
from collections import Counter, defaultdict
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ct_utils import window_uint8, preview_panel, resample, center_fit, make_drr  # noqa: E402


def process_hu(hu, spacing, cfg):
    """Resample to the fixed field of view, centre-fit to cfg.shape, and make the DRR (numpy/scipy/cv2 only)."""
    target = tuple(f / s for f, s in zip(cfg.fov, cfg.shape))
    fixed = center_fit(resample(hu, spacing, target), cfg.shape, fill=-1024.0)
    drr = make_drr(fixed, size=cfg.drr_px, spacing_zx=(target[0], target[2]))
    return fixed, drr, target


CFG = None
HEADER_FIELDS = ("PatientID", "StudyInstanceUID", "SeriesInstanceUID", "Modality", "SeriesDescription",
                 "StudyDescription", "ImageType", "SliceThickness", "ConvolutionKernel", "Manufacturer",
                 "ManufacturerModelName", "ContrastBolusAgent", "PixelSpacing", "ImageOrientationPatient",
                 "Rows", "Columns")


# --------------------------------------------------------------------------- DICOM access (pydicom only here)
def read_header(path):
    import pydicom
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    out = {}
    for f in HEADER_FIELDS:
        v = getattr(ds, f, None)
        if v is not None and f in ("ImageType", "PixelSpacing", "ImageOrientationPatient", "ConvolutionKernel"):
            v = list(v) if not isinstance(v, str) else [v]
        out[f] = v
    return out


def read_slices(paths):
    """-> list of (z_position, HU 2D array, header) for every image slice in a series."""
    import pydicom
    out = []
    for p in paths:
        ds = pydicom.dcmread(p, force=True)
        if not hasattr(ds, "ImagePositionPatient") or not hasattr(ds, "pixel_array"):
            continue
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        hu = ds.pixel_array.astype(np.float32) * slope + inter
        out.append((float(ds.ImagePositionPatient[2]), hu,
                    {"iop": [float(x) for x in ds.ImageOrientationPatient],
                     "ps": [float(x) for x in ds.PixelSpacing]},
                    np.array([float(x) for x in ds.ImagePositionPatient])))
    return out


# --------------------------------------------------------------------------- indexing and series choice
def list_series_dirs(root):
    """TCIA downloads keep one series per folder: return {folder: [dicom files]}."""
    out = {}
    for dirpath, _, files in os.walk(root, onerror=lambda e: print(f"[warn] cannot read {e.filename}")):
        dcm = sorted(os.path.join(dirpath, f) for f in files if f.lower().endswith(".dcm"))
        if dcm:
            out[dirpath] = dcm
    return out


def index_one(job):
    release, folder, files = job
    try:
        h = read_header(files[0])
        h.update(release=release, folder=folder, n_files=len(files))
        return h
    except Exception as e:
        return {"release": release, "folder": folder, "n_files": len(files), "error": repr(e)}


def is_axial(iop):
    if not iop or len(iop) != 6:
        return False
    n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
    return abs(n[2]) > 0.9


SHARP_KERNEL = re.compile(r"BONE|LUNG|DETAIL|EDGE|SHARP|^L$|^Y[A-D]$|^[BI][5-9]\d|^FC[5-9]\d", re.I)


def kernel_str(s):
    k = s.get("ConvolutionKernel")
    return "/".join(map(str, k)) if isinstance(k, list) else str(k or "")


def is_soft_kernel(s):
    return not SHARP_KERNEL.search(kernel_str(s).split("/")[0].strip())


def thickness(s):
    try:
        return float(s.get("SliceThickness") or 99)
    except (TypeError, ValueError):
        return 99.0


def choose_series(series, prefer_mm=3.0):
    """One series per study. Ranking, to keep 1a and 1b comparable: a standard/soft kernel first,
    then the slice thickness closest to prefer_mm, then the most slices."""
    cands = [s for s in series
             if s.get("Modality") == "CT" and is_axial(s.get("ImageOrientationPatient"))
             and not any(t in ("LOCALIZER", "SCOUT") for t in (s.get("ImageType") or []))
             and s["n_files"] >= 40]
    if not cands:
        return None, "no axial CT series with >= 40 slices"
    best = min(cands, key=lambda s: (0 if is_soft_kernel(s) else 1, round(abs(thickness(s) - prefer_mm), 2), -s["n_files"]))
    return best, "{} kernel, {:g} mm".format("soft" if is_soft_kernel(best) else "sharp (no soft option)", thickness(best))


# --------------------------------------------------------------------------- volume assembly
def assemble(slices):
    """Stack a series and orient it to our (z, y, x) RAS convention. Works for any axis-aligned
    acquisition (axial, coronal or sagittal reformats). DICOM directions are LPS."""
    # keep only slices sharing the series' most common size and orientation (drops scouts, dose reports)
    from collections import Counter as _C
    sig = lambda t: (t[1].shape, tuple(round(v, 3) for v in t[2]["iop"]))
    common, _ = _C(sig(t) for t in slices).most_common(1)[0]
    n_all = len(slices)
    slices = [t for t in slices if sig(t) == common]
    dropped = n_all - len(slices)
    if len(slices) < 10:
        raise ValueError(f"only {len(slices)} consistent slices (of {n_all})")
    iop, ps = slices[0][2]["iop"], slices[0][2]["ps"]
    r, c = np.array(iop[:3], float), np.array(iop[3:], float)      # r: +column direction, c: +row direction
    n = np.cross(r, c)                                              # slice normal
    pos = [float(np.dot(t[3], n)) for t in slices]
    order = np.argsort(pos)
    slices = [slices[i] for i in order]
    pos = np.array([pos[i] for i in order])
    keep = np.concatenate([[True], np.diff(pos) > 1e-3])           # drop duplicate positions
    slices = [s_ for s_, k in zip(slices, keep) if k]
    pos = pos[keep]
    d = np.diff(pos)
    ds = float(np.median(d))
    vol = np.stack([t[1] for t in slices]).astype(np.float32)      # (along n, rows, cols)
    dirs, spac = [n, c, r], [ds, float(ps[0]), float(ps[1])]
    axis_of, sign_of = [], []
    for v in dirs:
        a = int(np.argmax(np.abs(v)))
        if abs(v[a]) < 0.9:
            raise ValueError(f"oblique orientation {iop}")
        axis_of.append(a); sign_of.append(np.sign(v[a]))
    if sorted(axis_of) != [0, 1, 2]:
        raise ValueError(f"degenerate orientation {iop}")
    perm = [axis_of.index(2), axis_of.index(1), axis_of.index(0)]   # array axes for patient z, y, x
    vol = vol.transpose(perm)
    flips = []
    if sign_of[perm[0]] < 0:                                       # want z: inferior -> superior (+z in LPS)
        vol = vol[::-1]; flips.append("z")
    if sign_of[perm[1]] > 0:                                       # want y: posterior -> anterior (-y in LPS)
        vol = vol[:, ::-1]; flips.append("y")
    if sign_of[perm[2]] > 0:                                       # want x: patient left -> right (-x in LPS)
        vol = vol[:, :, ::-1]; flips.append("x")
    spacing = tuple(spac[a] for a in perm)
    plane = {2: "axial", 1: "coronal", 0: "sagittal"}[axis_of[0]]
    info = {"plane": plane, "slices": len(slices), "dropped_slices": dropped, "slice_spacing": round(ds, 3),
            "max_gap/median": round(float(d.max() / ds), 2) if len(d) else 1.0,
            "spacing_zyx": tuple(round(v, 3) for v in spacing), "flips": "".join(flips) or "none",
            "hu_p1_p99": tuple(round(float(v)) for v in np.percentile(vol[::4, ::4, ::4], (1, 99)))}
    return np.ascontiguousarray(vol), spacing, info


# --------------------------------------------------------------------------- per study (worker)
def init(cfg):
    global CFG
    CFG = cfg
    cv2.setNumThreads(0)


def work(job):
    vid, files = job
    vol_p = os.path.join(CFG.out, "volumes", vid + ".npy")
    drr_p = os.path.join(CFG.out, "drr", vid + ".png")
    if not CFG.inspect and os.path.exists(vol_p) and os.path.exists(drr_p):
        return vid, True, "exists"
    try:
        hu, spacing, info = assemble(read_slices(files))
        if info["max_gap/median"] > 1.6:
            info["warning"] = "uneven slice spacing (missing slices?)"
        if info["dropped_slices"]:
            info["warning"] = (info.get("warning", "") + f" dropped {info['dropped_slices']} slice(s) of another size/orientation").strip()
        fixed, drr, target = process_hu(hu, spacing, CFG)
        if CFG.inspect:
            cv2.imwrite(os.path.join(CFG.out, "inspect", vid + ".png"), preview_panel(fixed, drr, target))
            return vid, True, info
        np.save(vol_p, window_uint8(fixed))
        cv2.imwrite(drr_p, drr)
        return vid, True, info.get("warning", "ok")
    except Exception:
        return vid, False, traceback.format_exc(limit=3)


# --------------------------------------------------------------------------- splits, reports
LIST_RE = re.compile(r"MIDRC-RICORD-(1[AB])/ID(.+?)_date([\d-]+)_(\d+)\.\d+-(.*)-([^-]{5})\.nii\.gz$")


def read_unimiss_lists(folder):
    """RICORD_{train,val,test}.txt as used by UniMiSS+: '<...>/MIDRC-RICORD-1A/ID<patient>_date<..>_<series#>.<..>-<desc>-<uid tail>.nii.gz <label>'.
    Returns {(release, patient, uid_tail): (split, label, stem)}."""
    out = {}
    for sp in ("train", "val", "test"):
        fp = os.path.join(folder, f"RICORD_{sp}.txt")
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        for line in open(fp):
            tok = line.strip().split()
            if not tok:
                continue
            m = LIST_RE.search(tok[0])
            if not m:
                raise ValueError(f"unrecognised line in {fp}: {line.strip()}")
            rel, pid, _, _, _, tail = m.groups()
            stem = os.path.basename(tok[0])[:-len(".nii.gz")]
            out[(rel.lower(), pid, tail)] = (sp, int(tok[1]), stem)
    return out


def list_key(s):
    pid = str(s.get("PatientID") or "")
    pid = re.sub(r"^MIDRC-RICORD-1[AB]-", "", pid)
    return (s["release"], pid, str(s.get("SeriesInstanceUID") or "")[-5:])


def split_of(pid, label, fracs, seed):
    h = int(hashlib.md5(f"{seed}-{label}-{pid}".encode()).hexdigest(), 16) % 10000 / 10000.0
    return "train" if h < fracs[0] else ("val" if h < fracs[0] + fracs[1] else "test")


def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(s))


def balance_report(chosen, path):
    lines = ["RICORD 1a (COVID+) vs 1b (COVID-), chosen series only. Large differences = possible shortcut.\n"]
    for s in chosen:
        m = re.search(r"SITE\d+", str(s.get("PatientID")))
        s["Site"] = m.group(0) if m else "?"
        s["KernelType"] = "soft" if is_soft_kernel(s) else "sharp"
    for key in ("Site", "KernelType", "Manufacturer", "ManufacturerModelName", "ConvolutionKernel", "ContrastBolusAgent", "StudyDescription"):
        lines.append(f"== {key}")
        for rel in ("1a", "1b"):
            vals = Counter(str(s.get(key) if not isinstance(s.get(key), list) else "/".join(map(str, s.get(key))))
                           for s in chosen if s["release"] == rel)
            lines.append(f"  {rel}: " + ", ".join(f"{k} ({v})" for k, v in vals.most_common(6)))
    for key in ("SliceThickness", "n_files"):
        lines.append(f"== {key} (median, min-max)")
        for rel in ("1a", "1b"):
            v = []
            for s in chosen:
                if s["release"] == rel:
                    try:
                        v.append(float(s.get(key)))
                    except (TypeError, ValueError):
                        pass
            if v:
                lines.append(f"  {rel}: {np.median(v):.2f}  ({min(v):.2f}-{max(v):.2f})  n={len(v)}")
    open(path, "w").write("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_1a", required=True)
    p.add_argument("--root_1b", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--inspect", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--split", default="0.7,0.1,0.2", help="train,val,test fractions of patients (per label)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--unimiss_lists", default="", help="folder with RICORD_train/val/test.txt (UniMiSS+ splits)")
    p.add_argument("--samples", choices=["listed", "per_study"], default="listed",
                   help="with --unimiss_lists: every listed series (UniMiSS+ protocol) or one series per study")
    p.add_argument("--split_dir", default="", help="folder with train.txt / val.txt / test.txt of IDs (overrides --split)")
    p.add_argument("--prefer_mm", type=float, default=3.0, help="preferred slice thickness when choosing the series")
    p.add_argument("--fov", default="360,360,360")
    p.add_argument("--shape", default="160,256,256")
    p.add_argument("--drr_px", type=int, default=512)
    p.add_argument("--lists_only", action="store_true")
    cfg = p.parse_args()
    cfg.fov = tuple(float(v) for v in cfg.fov.split(","))
    cfg.shape = tuple(int(v) for v in cfg.shape.split(","))
    fracs = tuple(float(v) for v in cfg.split.split(","))
    for sub in ("volumes", "drr", "inspect", "lists"):
        os.makedirs(os.path.join(cfg.out, sub), exist_ok=True)

    # 1. index every series folder (first header of each)
    jobs = []
    for rel, root in (("1a", cfg.root_1a), ("1b", cfg.root_1b)):
        dirs = list_series_dirs(root)
        print(f"[index] {rel}: {len(dirs)} series folders, {sum(len(v) for v in dirs.values())} DICOM files under {root}")
        jobs += [(rel, d, f) for d, f in dirs.items()]
    with Pool(max(1, cfg.workers)) as pool:
        series = pool.map(index_one, jobs, chunksize=4)
    bad = [s for s in series if "error" in s]
    for s in bad[:5]:
        print(f"[warn] unreadable header in {s['folder']}: {s['error']}")
    series = [s for s in series if "error" not in s]
    files_of = {(rel, d): f for rel, d, f in jobs}

    if cfg.unimiss_lists:
        lists = read_unimiss_lists(cfg.unimiss_lists)
        by_key = {list_key(s): s for s in series}
        unmatched = [k for k in lists if k not in by_key]
        print(f"[lists] UniMiSS+ lists: {len(lists)} series listed; {len(lists) - len(unmatched)} matched to DICOM series, "
              f"{len(unmatched)} not found{': ' + str(unmatched[:5]) if unmatched else ''}")
        if cfg.samples == "listed":
            chosen = []
            for k, (sp, lab, stem) in lists.items():
                if k in by_key:
                    s = by_key[k]
                    s.update(vid=stem, label=lab, split=sp)
                    chosen.append(s)
            print(f"[select] protocol 'listed' (UniMiSS+): every listed series is one sample -> {len(chosen)} samples")
            run_selected(cfg, chosen, jobs)
            return
        patient_split = {}
        for (rel, pid, _), (sp, _, _) in lists.items():
            patient_split[(rel, pid)] = sp
    else:
        patient_split = None

    # 2. one series per study
    by_study = defaultdict(list)
    for s in series:
        by_study[(s["release"], s.get("StudyInstanceUID"))].append(s)
    chosen, rows = [], []
    for (rel, study), ss in sorted(by_study.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        best, why = choose_series(ss, cfg.prefer_mm)
        for s in ss:
            rows.append([rel, s.get("PatientID"), study, s.get("SeriesInstanceUID"), s.get("SeriesDescription"),
                         s.get("SliceThickness"), s["n_files"], "CHOSEN" if s is best else "", why if s is best else ""])
        if best is not None:
            best["vid"] = f"ricord{rel}_{safe(best.get('PatientID'))}_{hashlib.md5(str(study).encode()).hexdigest()[:8]}"
            best["label"] = 1 if rel == "1a" else 0
            chosen.append(best)
        else:
            print(f"[skip] {rel} study {study}: {why}")
    with open(os.path.join(cfg.out, "series_index.csv"), "w") as f:
        f.write("release,patient,study,series,description,slice_thickness,n_files,chosen,reason\n")
        for r in rows:
            f.write(",".join('"' + str(x).replace('"', "'") + '"' for x in r) + "\n")
    pid_rel = defaultdict(set)
    for s in chosen:
        pid_rel[s.get("PatientID")].add(s["release"])
    both = [pid for pid, rels in pid_rel.items() if len(rels) > 1]
    if both:
        print(f"[warn] {len(both)} patient IDs appear in both 1a and 1b: {both[:5]}")
    print(f"[select] {sum(s['release'] == '1a' for s in chosen)} studies from 1a, "
          f"{sum(s['release'] == '1b' for s in chosen)} from 1b "
          f"({len(series)} series indexed; details in series_index.csv)")

    if patient_split is not None:
        for s in chosen:
            s["split"] = patient_split.get((s["release"], list_key(s)[1]))
        missing = sum(s["split"] is None for s in chosen)
        print(f"[split] protocol 'per_study': UniMiSS+ patient splits; {missing} studies without a listed patient (left out)")
        chosen = [s for s in chosen if s["split"] is not None]
    elif cfg.split_dir:
        # external split: <split_dir>/{train,val,test}.txt, one ID per line (first token). An ID may be a
        # PatientID, StudyInstanceUID, SeriesInstanceUID or the output id; studies not listed are left out.
        ext = {}
        for sp in ("train", "val", "test"):
            fp = os.path.join(cfg.split_dir, sp + ".txt")
            if os.path.exists(fp):
                for line in open(fp):
                    tok = line.strip().split()
                    if tok:
                        ext[os.path.splitext(os.path.basename(tok[0]))[0]] = sp
        for s in chosen:
            s["split"] = next((ext[k] for k in (s.get("PatientID"), s.get("StudyInstanceUID"),
                                                s.get("SeriesInstanceUID"), s["vid"]) if k in ext), None)
        missing = sum(s["split"] is None for s in chosen)
        print(f"[split] from {cfg.split_dir}: {len(ext)} ids listed; {len(chosen) - missing} studies matched, {missing} not listed (left out)")
        chosen = [s for s in chosen if s["split"] is not None]
    else:
        for s in chosen:
            s["split"] = split_of(s.get("PatientID"), s["label"], fracs, cfg.seed)
        print(f"[split] patient-level, stratified by label, fractions {fracs} (seed {cfg.seed}); --split_dir for external splits")
    run_selected(cfg, chosen, jobs)


def run_selected(cfg, chosen, jobs):
    """Process the selected samples (or a few with --inspect), write lists and the yaml block."""
    files_of = {(rel, d): f for rel, d, f in jobs}
    print(balance_report(chosen, os.path.join(cfg.out, "balance_report.txt")))
    if cfg.inspect:
        pick = [s for s in chosen if s["release"] == "1a"][:cfg.inspect // 2] + \
               [s for s in chosen if s["release"] == "1b"][:cfg.inspect - cfg.inspect // 2]
    else:
        pick = chosen
    if not cfg.lists_only:
        todo = [(s["vid"], files_of[(s["release"], s["folder"])]) for s in pick]
        n_ok, failed = 0, []
        with Pool(max(1, min(cfg.workers, len(todo))), initializer=init, initargs=(cfg,)) as pool:
            for i, (vid, ok, info) in enumerate(pool.imap_unordered(work, todo, chunksize=1)):
                if ok:
                    n_ok += 1
                    if cfg.inspect:
                        print(f"\n{vid}:")
                        for k, v in info.items():
                            print(f"   {k:15s} {v}")
                    elif info not in ("ok", "exists"):
                        print(f"[note] {vid}: {info}")
                else:
                    failed.append(vid)
                    print(f"[FAIL] {vid}\n{info}")
                if not cfg.inspect and (i + 1) % 25 == 0:
                    print(f"[progress] {i + 1}/{len(todo)} ok={n_ok} failed={len(failed)}", flush=True)
        if cfg.inspect:
            print(f"\nPreviews in {os.path.join(cfg.out, 'inspect')}: axial = sternum at top, heart on the image right; "
                  "coronal = head at top; DRR = frontal CXR look.")
            return
        print(f"[done] ok={n_ok} failed={len(failed)}")

    out = os.path.abspath(cfg.out)
    counts = {}
    for sp in ("train", "val", "test"):
        rows_sp = sorted((s for s in chosen if s["split"] == sp and os.path.exists(os.path.join(out, "volumes", s["vid"] + ".npy"))),
                         key=lambda s: s["vid"])
        with open(os.path.join(out, "lists", f"{sp}.txt"), "w") as f:
            for s in rows_sp:
                f.write(f"volumes/{s['vid']}.npy {s['label']}\n")
        counts[sp] = f"{len(rows_sp)} samples ({sum(s['label'] for s in rows_sp)} COVID+)"
    print(f"[lists] {counts} -> {os.path.join(out, 'lists')}")
    print(f"""
# ---- add to datasets_config.yaml ----
RICORD:
  loader: CTVolume
  dims: 3
  data_dir: {out}
  train_list: {out}/lists/train.txt
  val_list: {out}/lists/val.txt
  test_list: {out}/lists/test.txt
  diseases: ['COVID-19']
  task_type: binary classification
  batch_size: 8
  accum_steps: 1
  drr_head: false                       # true only for 2D+3D runs
  dataset_kwargs:
    volume_size: [128, 224, 224]
    drr_dir: drr
    drr_size: 224
    return_drr: false                   # true only for 2D+3D runs
""")



if __name__ == "__main__":
    main()
