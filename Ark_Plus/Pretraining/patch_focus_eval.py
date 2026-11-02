"""
patch_focus_eval.py -- run from Ark_Plus/Pretraining/ (branch switchable-3d):   python patch_focus_eval.py
Adds --focus_eval True: in CYCLIC training, at every test epoch (epoch % test_epoch == 0, and the last epoch),
right after EACH dataset's training phase, every dataset's test set is evaluated with the student and the
teacher. Rows go to Outputs/.../<exp>_<run>_focus.csv:
    epoch, after_phase, eval_dataset, state (focused = the dataset just trained / unfocused), student_mAUC, teacher_mAUC
Training is unchanged (evaluation only). Ignored when --interleave is on (there are no phases).
Makes .bak_focus backups; refuses to write anything that does not compile; does nothing if already applied.
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
    shutil.copy2(path, path + ".bak_focus")
    open(path, "w", newline="\n").write(src)
    print("patched:", path)

patch("main_ark.py", [
    (r'^([ \t]*)parser\.add_option\("--interleave"',
     r'\1parser.add_option("--focus_eval", dest="focus_eval", help="cyclic runs: at test epochs, test every dataset after each dataset phase (focused/unfocused)", default=False, action="callback",' "\n"
     r'\1                  callback=vararg_callback_bool)' "\n" r'\g<0>',
     '--interleave option (apply patch_interleave.py first)'),
], "--focus_eval")

FOCUS = r'''\g<0>\1if getattr(args, 'focus_eval', False) and (epoch % args.test_epoch == 0 or epoch + 1 == args.pretrain_epochs):
\1    _focus_file = os.path.join(output_path, exp + "_" + args.exp_name + "_focus.csv")
\1    _new_file = not os.path.exists(_focus_file)
\1    with open(_focus_file, 'a') as _ff:
\1        if _new_file:
\1            _ff.write("epoch,after_phase,eval_dataset,state,student_mAUC,teacher_mAUC\\n")
\1        for _j, _ds in enumerate(dataset_list):
\1            _mc = datasets_config[_ds]['task_type'] == "multi-class classification"
\1            _ys, _ps = test_classification(model, _j, data_loader_list_test[_j], device, _mc, is_3d=is_3d_list[_j])
\1            _yt, _pt = test_classification(teacher, _j, data_loader_list_test[_j], device, _mc, is_3d=is_3d_list[_j])
\1            if _ds == "CheXpert":
\1                _dis = datasets_config[_ds]['diseases']
\1                _idx = [_dis.index(c) for c in datasets_config['CheXpert']['test_diseases_name']]
\1                _ys, _ps, _yt, _pt = _ys[:, _idx], _ps[:, _idx], _yt[:, _idx], _pt[:, _idx]
\1            _s = float(np.mean(metric_AUROC(_ys, _ps, _ys.shape[1])))
\1            _t = float(np.mean(metric_AUROC(_yt, _pt, _yt.shape[1])))
\1            _ff.write("{},{},{},{},{:.4f},{:.4f}\\n".format(epoch, dataset_list[i], _ds, "focused" if _j == i else "unfocused", _s, _t))
\1    print("[focus_eval] epoch {:04d}: tested all datasets after the {} phase".format(epoch, dataset_list[i]))
\1    model.train()
'''
patch("engine.py", [
    (r"^([ \t]*)it \+= 1\n", FOCUS, "the 'it += 1' line of the cyclic loop"),
], "focus_eval")
