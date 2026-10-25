"""
patch_modality_lr.py  -- run from Ark_Plus/Pretraining/ (branch switchable-3d):  python patch_modality_lr.py
Adds to the unified Ark+ code:
  1. --lr_scale_3d S  (main_ark.py + engine.py): CT (dims 3) datasets train at S x the scheduled learning rate,
     X-ray datasets at 1 x.  Default 1.0 = unchanged behaviour.  Example: --lr 0.01 --lr_scale_3d 0.03 gives
     0.01 for X-rays and 0.0003 for CT, both following the same warm-up + cosine shape.
  2. NaN guard (engine.py): if any validation loss is not finite, the run stops with a clear message BEFORE the
     checkpoint is overwritten, so the last good checkpoint survives (no wasted epochs, no crash at test time).
Makes .bak_modlr backups; refuses to write anything that does not compile; does nothing if already applied.
"""
import re, shutil, sys

def patch(path, edits):
    src = open(path, newline="").read().replace("\r\n", "\n")
    if "lr_scale_3d" in src:
        print("already patched:", path); return
    for pattern, repl, what in edits:
        new, n = re.subn(pattern, repl, src, count=1, flags=re.M)
        if n != 1:
            sys.exit(f"could not find in {path}: {what}")
        src = new
    compile(src, path, "exec")
    shutil.copy2(path, path + ".bak_modlr")
    open(path, "w", newline="\n").write(src)
    print("patched:", path)

patch("main_ark.py", [
    (r'^([ \t]*)parser\.add_option\("--w_xdim".*\n',
     r'\g<0>\1parser.add_option("--lr_scale_3d", dest="lr_scale_3d", help="CT datasets train at this multiple of the scheduled lr (X-rays at 1x)", default=1.0, type="float")' "\n",
     '--w_xdim option'),
])
patch("engine.py", [
    (r"^([ \t]*)print\('Epoch \{:04d\}: lr = \{:\.6g\}'\.format\(epoch, current_lr\)\)\n",
     r"\g<0>\1if args.lr_scale_3d != 1.0 and any(is_3d_list):" "\n"
     r"\1    print('Epoch {:04d}: lr for CT datasets = {:.6g} (x{:g})'.format(epoch, current_lr * args.lr_scale_3d, args.lr_scale_3d))" "\n",
     "per-epoch lr print"),
    (r"^([ \t]*)train_one_epoch\(model, i, dataset_list\[i\], data_loader,",
     r"\1for _g in optimizer.param_groups:   # per-modality learning rate" "\n"
     r"\1    _g['lr'] = current_lr * (args.lr_scale_3d if is_3d_list[i] else 1.0)" "\n"
     r"\g<0>",
     "train_one_epoch call"),
    (r"^([ \t]*)avg_val_loss = np\.average\(val_loss_list\)\n",
     r"\1if not np.all(np.isfinite(val_loss_list)):   # NaN guard: stop before overwriting the last good checkpoint" "\n"
     r"\1    _msg = 'STOPPED at epoch {:04d}: non-finite validation loss {} for {} (training diverged; last good checkpoint kept)'.format(epoch, val_loss_list, dataset_list)" "\n"
     r"\1    print(_msg)" "\n"
     r"\1    with open(log_file, 'a') as _log:" "\n"
     r"\1        _log.write(_msg + '\\n')" "\n"
     r"\1    raise RuntimeError(_msg)" "\n"
     r"\g<0>",
     "avg_val_loss line"),
])
