"""
patch_interleave.py -- run from Ark_Plus/Pretraining/ (branch switchable-3d) AFTER patch_ct_clip.py:
    python patch_interleave.py
Adds --interleave True for mixed 2D+3D runs. Instead of training one whole dataset after another each epoch
(Ark's cyclic schedule: e.g. 75 VinDr batches, then 25 RICORD batches), the datasets' batches are mixed
through the epoch in proportion to their sizes (V V V R V V V R ...). Each dataset still sees each of its
batches exactly once per epoch, with the same loss, optimizer (CT steps use optimizer_3d), learning rates
and clipping. The teacher is updated exactly as in the cyclic schedule: once per dataset at the end of the
epoch, with that dataset's momentum. Runs without --interleave, and single-modality runs, are unchanged.
Patches trainer.py, engine.py, main_ark.py (.bak_interleave backups); refuses to write anything that does not
compile; does nothing if already applied.
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
    shutil.copy2(path, path + ".bak_interleave")
    open(path, "w", newline="\n").write(src)
    print("patched:", path)

if "clip_grad_3d" not in open("engine.py").read():
    sys.exit("apply patch_modality_lr.py, patch_modality_optim.py and patch_ct_clip.py first")

INTERLEAVED = '''def train_one_epoch_interleaved(model, tasks, device, epoch, ema_mode, teacher, momentum_schedule, it,
                                w_xdim=0.5, use_amp=False):
    """Mixed 2D+3D training with the datasets' batches interleaved through the epoch in proportion to their
    sizes (each dataset still sees each of its batches once). tasks: list of dicts with keys
    head, name, loader, criterion, optimizer, drr_head_n. Per batch: the same loss as train_one_epoch
    (classification + teacher consistency [+ DRR terms]), one optimizer step (accum_steps 1).
    Teacher EMA as in the cyclic schedule: one update per dataset at the end of the epoch, momentum[it + k]."""
    assert ema_mode == "epoch", "--interleave supports --ema_mode epoch only"
    MSE = torch.nn.MSELoss()
    model.train()
    n = [len(t["loader"]) for t in tasks]
    done = [0] * len(tasks)
    iters = [iter(t["loader"]) for t in tasks]
    coffs = [(momentum_schedule[it + k] - 0.9) * 5 for k in range(len(tasks))]
    meters = []
    for t, nk in zip(tasks, n):
        bt = MetricLogger('Time', ':6.3f')
        lc = MetricLogger('Loss_' + t["name"] + ' cls', ':.4e')
        lm = MetricLogger('Loss_' + t["name"] + ' mse', ':.4e')
        ms, lx = [bt, lc, lm], None
        if t["drr_head_n"] is not None:
            lx = MetricLogger('Loss_' + t["name"] + ' xdim', ':.4e')
            ms.append(lx)
        meters.append((bt, lc, lm, lx, ProgressLogger(nk, ms, prefix="Epoch: [{}]".format(epoch))))
    end = time.time()
    for _ in range(sum(n)):
        # next batch comes from the dataset that is furthest behind its share of the epoch
        k = min((j for j in range(len(tasks)) if done[j] < n[j]), key=lambda j: (done[j] + 0.5) / n[j])
        t = tasks[k]
        bt, lc, lm, lx, progress = meters[k]
        batch = next(iters[k])
        samples1, samples2, targets = batch[0].float().to(device), batch[1].float().to(device), batch[2].float().to(device)
        use_drr = t["drr_head_n"] is not None and len(batch) > 3
        coff = coffs[k]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            with torch.no_grad():
                feat_t, pred_t = teacher(samples2, t["head"])
            feat_s, pred_s = model(samples1, t["head"])
            loss_cls = t["criterion"](pred_s.float(), targets)
            loss_const = MSE(feat_s.float(), feat_t.float())
            loss = (1 - coff) * loss_cls + coff * loss_const
            if use_drr:
                drr = batch[3].float().to(device)
                with torch.no_grad():
                    feat_dt, _ = teacher(drr, t["drr_head_n"])
                feat_ds, pred_ds = model(drr, t["drr_head_n"])
                loss_drr_cls = t["criterion"](pred_ds.float(), targets)
                loss_drr_const = MSE(feat_ds.float(), feat_dt.float())
                loss_xdim = MSE(feat_s.float(), feat_dt.float()) + MSE(feat_ds.float(), feat_t.float())
                loss = loss + (1 - coff) * loss_drr_cls + coff * (loss_drr_const + w_xdim * loss_xdim)
        t["optimizer"].zero_grad()
        loss.backward()
        t["optimizer"].step()
        lc.update(loss_cls.item(), samples1.size(0))
        lm.update(loss_const.item(), samples1.size(0))
        if use_drr:
            lx.update(loss_xdim.item(), samples1.size(0))
        bt.update(time.time() - end)
        end = time.time()
        if done[k] % 50 == 0:
            progress.display(done[k])
        done[k] += 1
    for k in range(len(tasks)):
        ema_update_teacher(model, teacher, momentum_schedule, it + k)


'''
patch("trainer.py", [
    (r"^def ema_update_teacher\(", INTERLEAVED.replace("\\", "\\\\") + r"def ema_update_teacher(", "def ema_update_teacher("),
], "train_one_epoch_interleaved")

patch("main_ark.py", [
    (r'^([ \t]*)parser\.add_option\("--clip_grad_3d".*\n',
     r'\g<0>\1parser.add_option("--interleave", dest="interleave", help="mixed 2D+3D runs: interleave the datasets\' batches through each epoch instead of one dataset after another", default=False, action="callback",' "\n"
     r'\1                  callback=vararg_callback_bool)' "\n",
     "--clip_grad_3d option"),
], "--interleave")

patch("engine.py", [
    (r"^from trainer import train_one_epoch, test_classification, evaluate\n",
     "from trainer import train_one_epoch, test_classification, evaluate, train_one_epoch_interleaved\n",
     "from trainer import ..."),
    (r"^([ \t]*)set_pretrained_frozen\(warmup_3d\)\n",
     r"\g<0>"
     r"\1_interleave = bool(getattr(args, 'interleave', False)) and any(is_3d_list) and not all(is_3d_list) and not warmup_3d" "\n"
     r"\1if _interleave:   # mixed 2D+3D: batches of all datasets interleaved through the epoch" "\n"
     r"\1    _tasks = []" "\n"
     r"\1    for _k, _dl in enumerate(data_loader_list_train):" "\n"
     r"\1        if datasets_config[dataset_list[_k]].get('accum_steps', 1) != 1:" "\n"
     r"\1            raise RuntimeError('--interleave supports accum_steps 1 only ({})'.format(dataset_list[_k]))" "\n"
     r"\1        _o = optimizer_3d if (optimizer_3d is not None and is_3d_list[_k]) else optimizer" "\n"
     r"\1        for _g in _o.param_groups:" "\n"
     r"\1            _g['lr'] = current_lr * (args.lr_scale_3d if is_3d_list[_k] else 1.0)" "\n"
     r"\1        _crit = torch.nn.CrossEntropyLoss() if datasets_config[dataset_list[_k]]['task_type'] == 'multi-class classification' else torch.nn.BCEWithLogitsLoss()" "\n"
     r"\1        _tasks.append(dict(head=_k, name=dataset_list[_k], loader=_dl, criterion=_crit, optimizer=_o, drr_head_n=drr_head_index.get(_k)))" "\n"
     r"\1    if epoch == start_epoch:" "\n"
     r"\1        print('[interleave] mixing {} batches per epoch: {}'.format(sum(len(t['loader']) for t in _tasks), {t['name']: len(t['loader']) for t in _tasks}))" "\n"
     r"\1    train_one_epoch_interleaved(model, _tasks, device, epoch, args.ema_mode, teacher, momentum_schedule, it," "\n"
     r"\1                                w_xdim=args.w_xdim, use_amp=args.amp)" "\n"
     r"\1    it += len(dataset_list)" "\n",
     "set_pretrained_frozen(warmup_3d)"),
    (r"^([ \t]*)for i, data_loader in enumerate\(data_loader_list_train\):[ \t]*\n",
     r"\1for i, data_loader in enumerate([] if _interleave else data_loader_list_train):" "\n",
     "the cyclic training loop"),
], "train_one_epoch_interleaved")
