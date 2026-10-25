"""
patch_modality_optim.py  -- run from Ark_Plus/Pretraining/ (branch switchable-3d) AFTER patch_modality_lr.py:
    python patch_modality_optim.py
Fix for mixed 2D+3D runs: one shared SGD momentum buffer carried CT gradients into the following X-ray steps,
where they were applied at the X-ray learning rate (~30x the CT rate) and could wreck the shared encoder.
Now, when a run mixes X-ray and CT datasets with --lr_scale_3d != 1, CT steps use their OWN optimizer
(own momentum buffer, CT learning rate); X-ray steps use the original optimizer. Both update the same model.
Single-modality runs, and runs with --lr_scale_3d 1, are unchanged (only the original optimizer is used).
Both optimizer states are saved in checkpoints and restored on resume. Makes .bak_modopt backups;
refuses to write anything that does not compile; does nothing if already applied.
"""
import re, shutil, sys

path = "engine.py"
src = open(path, newline="").read().replace("\r\n", "\n")
if "optimizer_3d" in src:
    sys.exit("already patched: " + path)
if "lr_scale_3d" not in src:
    sys.exit("apply patch_modality_lr.py first")

def sub(pattern, repl, text, what, count=1, expect=1):
    new, n = re.subn(pattern, repl, text, count=count, flags=re.M)
    if n != expect:
        sys.exit(f"could not find (found {n}, expected {expect}): {what}")
    return new

# 1. a second optimizer for CT steps, only for mixed runs with a CT lr scale
src = sub(r"^([ \t]*)optimizer = create_optimizer\(args, model\)\n",
          r"\g<0>\1optimizer_3d = (create_optimizer(args, model)   # own momentum buffer for CT steps in mixed 2D+3D runs" "\n"
          r"\1                if (args.lr_scale_3d != 1.0 and any(is_3d_list) and not all(is_3d_list)) else None)" "\n"
          r"\1if optimizer_3d is not None:" "\n"
          r"\1    print('[modality optimizers] X-ray steps: optimizer (lr x1); CT steps: optimizer_3d (lr x{:g}), separate momentum'.format(args.lr_scale_3d))" "\n",
          src, "optimizer = create_optimizer(args, model)")
# 2. choose the optimizer per dataset phase, set its lr, pass it to train_one_epoch
src = sub(r"^([ \t]*)for _g in optimizer\.param_groups:   # per-modality learning rate\n",
          r"\1_opt = optimizer_3d if (optimizer_3d is not None and is_3d_list[i]) else optimizer" "\n"
          r"\1for _g in _opt.param_groups:   # per-modality learning rate (and, in mixed runs, per-modality optimizer)" "\n",
          src, "per-modality lr loop")
src = sub(r"^([ \t]*train_one_epoch\(model, i, dataset_list\[i\], data_loader, device, criterion, )optimizer(, epoch,)",
          r"\1_opt\2", src, "train_one_epoch(... optimizer ...)")
# 3. save both optimizer states (both save_checkpoint calls)
src = sub(r"^([ \t]*)'optimizer': optimizer\.state_dict\(\),\n",
          r"\g<0>\1'optimizer_3d': (optimizer_3d.state_dict() if optimizer_3d is not None else None)," "\n",
          src, "'optimizer': optimizer.state_dict(), (x2)", count=0, expect=2)
# 4. restore it on resume
src = sub(r"^([ \t]*)optimizer\.load_state_dict\(checkpoint\['optimizer'\]\)\n",
          r"\g<0>\1if optimizer_3d is not None and checkpoint.get('optimizer_3d'):" "\n"
          r"\1    optimizer_3d.load_state_dict(checkpoint['optimizer_3d'])" "\n",
          src, "optimizer.load_state_dict(checkpoint['optimizer'])")
compile(src, path, "exec")
shutil.copy2(path, path + ".bak_modopt")
open(path, "w", newline="\n").write(src)
print("patched:", path)
