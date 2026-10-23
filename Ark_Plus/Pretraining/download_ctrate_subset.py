"""
download_ctrate_subset.py  (needs: pip install --user huggingface_hub)

Downloads a reproducible subset of CT-RATE from Hugging Face: N randomly chosen training patients
(one reconstruction per scan), the validation set (all of it by default, used as the test set), and
the metadata + label CSVs. Uses the header-fixed volumes (dataset/train_fixed, dataset/valid_fixed).

One-time access: log in on huggingface.co, open the CT-RATE dataset page and accept its terms,
create a read token (Settings -> Access Tokens), then on Sol:  huggingface-cli login

  # 1. dry run: lists the selection and its total size, downloads nothing
  python download_ctrate_subset.py --out /scratch/ztaiyab/data/CT-RATE --n_train_patients 2000 --dry_run
  # 2. download (resumable: rerun the same command after an interruption)
  python download_ctrate_subset.py --out /scratch/ztaiyab/data/CT-RATE --n_train_patients 2000

Then preprocess with prep_ctrate.py --root /scratch/ztaiyab/data/CT-RATE ...
"""
import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "ibrahimhamamci/CT-RATE"
NAME_RE = re.compile(r"(train|valid)_(\d+)_([a-z]+)_(\d+)\.nii\.gz$")


def list_files(api, path, cache):
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    print(f"[list] listing {path} on the hub (a few minutes the first time)...", flush=True)
    files = [{"path": e.path, "size": getattr(e, "size", 0) or 0}
             for e in api.list_repo_tree(REPO, path_in_repo=path, recursive=True, repo_type="dataset")
             if e.path.endswith(".nii.gz")]
    with open(cache, "w") as f:
        json.dump(files, f)
    return files


def select(files, n_patients, seed):
    """One reconstruction (lowest id) per scan, for n_patients randomly chosen patients (0 = all)."""
    best = {}
    for f in files:
        m = NAME_RE.search(f["path"])
        if not m:
            continue
        split, pid, scan, rec = m.groups()
        key = (pid, scan)
        if key not in best or int(rec) < best[key][0]:
            best[key] = (int(rec), f)
    patients = sorted({pid for pid, _ in best}, key=int)
    if n_patients and n_patients < len(patients):
        rng = random.Random(seed)
        patients = sorted(rng.sample(patients, n_patients), key=int)
    keep = set(patients)
    chosen = [f for (pid, _), (_, f) in sorted(best.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])) if pid in keep]
    return chosen, len(keep)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n_train_patients", type=int, default=2000)
    p.add_argument("--n_valid_patients", type=int, default=0, help="0 = whole validation set (recommended: it is the test set)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--folders", default="train_fixed,valid_fixed")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub missing: pip install --user huggingface_hub")
    api = HfApi()
    os.makedirs(a.out, exist_ok=True)
    tr_dir, va_dir = a.folders.split(",")

    try:
        tr = list_files(api, f"dataset/{tr_dir}", os.path.join(a.out, f"_listing_{tr_dir}.json"))
        va = list_files(api, f"dataset/{va_dir}", os.path.join(a.out, f"_listing_{va_dir}.json"))
    except Exception as e:
        sys.exit(f"[error] could not list the repo ({e}).\nHave you accepted the CT-RATE terms on huggingface.co "
                 "and run `huggingface-cli login` with a read token?")
    tr_sel, n_tr = select(tr, a.n_train_patients, a.seed)
    va_sel, n_va = select(va, a.n_valid_patients, a.seed)

    tables = [e.path for e in api.list_repo_tree(REPO, path_in_repo="dataset", recursive=True, repo_type="dataset")
              if e.path.endswith(".csv") and ("metadata" in e.path or "abnormality" in e.path)]

    gb = lambda fs: sum(f["size"] for f in fs) / 1e9
    print(f"[select] train: {n_tr} patients, {len(tr_sel)} volumes, {gb(tr_sel):.0f} GB")
    print(f"[select] valid: {n_va} patients, {len(va_sel)} volumes, {gb(va_sel):.0f} GB")
    print(f"[select] tables: {tables}")
    print(f"[select] total download ~{gb(tr_sel) + gb(va_sel):.0f} GB into {a.out}")
    with open(os.path.join(a.out, "selection.json"), "w") as f:
        json.dump({"seed": a.seed, "n_train_patients": n_tr, "n_valid_patients": n_va,
                   "files": [x["path"] for x in tr_sel + va_sel]}, f)
    if a.dry_run:
        print("[dry run] nothing downloaded; selection saved to selection.json")
        return

    todo = tables + [x["path"] for x in tr_sel + va_sel]
    done = failed = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(hf_hub_download, REPO, path, repo_type="dataset", local_dir=a.out): path for path in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
                done += 1
            except Exception as e:
                failed += 1
                print(f"[FAIL] {futs[fut]}: {e}", flush=True)
            if (done + failed) % 50 == 0:
                print(f"[download] {done + failed}/{len(todo)} (failed {failed})", flush=True)
    print(f"[done] {done} files, {failed} failed. Rerun the same command to retry failures.")


if __name__ == "__main__":
    main()
