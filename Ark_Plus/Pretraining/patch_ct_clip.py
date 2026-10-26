"""
patch_ct_clip.py -- run from Ark_Plus/Pretraining/ (branch switchable-3d) AFTER patch_modality_optim.py:
    python patch_ct_clip.py
Adds --clip_grad_3d MAX (default 0 = off). In mixed 2D+3D runs (where CT steps have their own optimizer_3d),
every CT update first clips the gradient's total norm to MAX. Normal steps (norm below MAX) are untouched;
only spikes are scaled down. Each epoch logs the median and largest CT gradient norm and how many CT steps
were clipped, so you can see whether it ever acts. X-ray steps and single-modality runs are unchanged.
Makes .bak_ctclip backups; refuses to write anything that does not compile; does nothing if already applied.
"""
import re, shutil, sys

def patch(path, edits, marker):
    src = open(path, newline="").read().replace("\r\n", "\n")
    if marker in src:
        print("already patched:", path); return
    for pattern, repl, what in edits:
        new, n = re.subn(pattern, repl, src, count=1, flags=re.M)
        if n != 1:
            sys.exit(f"could not find in {path}: {what}")
        src = new
    compile(src, path, "exec")
    shutil.copy2(path, path + ".bak_ctclip")
    open(path, "w", newline="\n").write(src)
    print("patched:", path)

patch("main_ark.py", [
    (r'^([ \t]*)parser\.add_option\("--lr_scale_3d".*\n',
     r'\g<0>\1parser.add_option("--clip_grad_3d", dest="clip_grad_3d", help="mixed 2D+3D runs: clip the gradient norm of CT updates to this value (0 = off)", default=0.0, type="float")' "\n",
     "--lr_scale_3d option (apply patch_modality_lr.py first)"),
], "clip_grad_3d")

src = open("engine.py").read()
if "optimizer_3d" not in src:
    sys.exit("apply patch_modality_optim.py first")
patch("engine.py", [
    (r"^([ \t]*)if optimizer_3d is not None:\n([ \t]*)print\('\[modality optimizers\].*\n",
     r"\g<0>"
     r"\1_ct_clip = {'n': 0, 'clipped': 0, 'norms': []}" "\n"
     r"\1if optimizer_3d is not None and args.clip_grad_3d > 0:   # clip CT update spikes" "\n"
     r"\1    _ct_step = optimizer_3d.step" "\n"
     r"\1    def _ct_clipped_step(*a, **k):" "\n"
     r"\1        _nrm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_3d))" "\n"
     r"\1        _ct_clip['n'] += 1" "\n"
     r"\1        _ct_clip['norms'].append(_nrm)" "\n"
     r"\1        _ct_clip['clipped'] += int(_nrm > args.clip_grad_3d)" "\n"
     r"\1        return _ct_step(*a, **k)" "\n"
     r"\1    optimizer_3d.step = _ct_clipped_step" "\n"
     r"\1    print('[modality optimizers] CT updates: gradient norm clipped to {:g}'.format(args.clip_grad_3d))" "\n",
     "the [modality optimizers] print block"),
    (r"^([ \t]*)val_loss_list = \[\]\n",
     r"\1if _ct_clip['n']:" "\n"
     r"\1    _m = 'Epoch {:04d}: CT grad norm median {:.3g}, max {:.3g}; clipped {}/{} CT steps (max_norm {:g})'.format(" "\n"
     r"\1        epoch, float(np.median(_ct_clip['norms'])), max(_ct_clip['norms']), _ct_clip['clipped'], _ct_clip['n'], args.clip_grad_3d)" "\n"
     r"\1    print(_m)" "\n"
     r"\1    with open(log_file, 'a') as _log:" "\n"
     r"\1        _log.write(_m + '\\n')" "\n"
     r"\1    _ct_clip.update(n=0, clipped=0, norms=[])" "\n"
     r"\g<0>",
     "val_loss_list = []"),
], "clip_grad_3d")
