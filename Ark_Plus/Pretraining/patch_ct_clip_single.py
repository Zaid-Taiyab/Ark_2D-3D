"""
patch_ct_clip_single.py -- run from Ark_Plus/Pretraining/ AFTER patch_ct_clip.py:   python patch_ct_clip_single.py
Lets --clip_grad_3d also act in CT-only runs (every dataset is 3D, one optimizer), so a CT-only control can be
trained with exactly the same clipping as the CT steps of the mixed 2D+3D runs. Mixed runs are unchanged
(they still clip only optimizer_3d), X-ray-only runs are unchanged, and nothing changes when --clip_grad_3d is 0.
Makes a .bak_ctclip1 backup; refuses to write anything that does not compile; does nothing if already applied.
"""
import re, shutil, sys
path = "engine.py"
src = open(path, newline="").read().replace("\r\n", "\n")
if "_ct_opt" in src:
    sys.exit("already patched: " + path)
if "clip_grad_3d" not in src:
    sys.exit("apply patch_ct_clip.py first")
edits = [
    (r"^([ \t]*)if optimizer_3d is not None and args\.clip_grad_3d > 0:   # clip CT update spikes\n",
     r"\1_ct_opt = optimizer_3d if optimizer_3d is not None else (optimizer if all(is_3d_list) else None)   # CT-only runs: the one optimizer" "\n"
     r"\1if _ct_opt is not None and args.clip_grad_3d > 0:   # clip CT updates" "\n"),
    (r"^([ \t]*)_ct_step = optimizer_3d\.step\n", r"\1_ct_step = _ct_opt.step" "\n"),
    (r"^([ \t]*)optimizer_3d\.step = _ct_clipped_step\n", r"\1_ct_opt.step = _ct_clipped_step" "\n"),
]
for pattern, repl in edits:
    src, n = re.subn(pattern, repl, src, count=1, flags=re.M)
    if n != 1:
        sys.exit("could not find: " + pattern)
compile(src, path, "exec")
shutil.copy2(path, path + ".bak_ctclip1")
open(path, "w", newline="\n").write(src)
print("patched:", path)
