from utils import MetricLogger, ProgressLogger, save_image, save_snapshot
import time
import torch
from tqdm import tqdm
import wandb

def _preview(x):
    """First sample as HxWx3 numpy for save_image; 3D volumes -> middle axial slice."""
    x = x[0].detach().float().cpu()
    if x.dim() == 4:                      # (C, D, H, W)
        x = x[:, x.shape[1] // 2]
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    return x.numpy().transpose(1, 2, 0)


def train_one_epoch(model, use_head_n, dataset, data_loader_train, device, criterion, optimizer, epoch, ema_mode, teacher, momentum_schedule, it,
                    drr_head_n=None, w_xdim=0.5, accum_steps=1, use_amp=False):
    batch_time = MetricLogger('Time', ':6.3f')
    losses_cls = MetricLogger('Loss_'+dataset+' cls', ':.4e')
    losses_mse = MetricLogger('Loss_'+dataset+' mse', ':.4e')
    meters = [batch_time, losses_cls, losses_mse]
    if drr_head_n is not None:
        losses_xdim = MetricLogger('Loss_'+dataset+' xdim', ':.4e')
        meters.append(losses_xdim)
    progress = ProgressLogger(
        len(data_loader_train),
        meters,
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    MSE = torch.nn.MSELoss()
    coff = (momentum_schedule[it] - 0.9) * 5
    #print(momentum_schedule[it],it, coff)
    end = time.time()
    n_batches = len(data_loader_train)
    optimizer.zero_grad()
    for i, batch in enumerate(data_loader_train):
        # 2D datasets yield (student, teacher, target); CTVolume yields (student, teacher, target, drr)
        samples1, samples2, targets = batch[0].float().to(device), batch[1].float().to(device), batch[2].float().to(device)
        use_drr = drr_head_n is not None and len(batch) > 3

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            with torch.no_grad():
                feat_t, pred_t = teacher(samples2, use_head_n)
            feat_s, pred_s = model(samples1, use_head_n)
            loss_cls = criterion(pred_s.float(), targets)
            loss_const = MSE(feat_s.float(), feat_t.float())
            loss = (1-coff) * loss_cls + coff * loss_const

            if use_drr:
                # CT <-> DRR: the DRR inherits the CT labels (own head), and the projected embeddings
                # of the volume and its DRR are pulled together across student/teacher.
                drr = batch[3].float().to(device)
                with torch.no_grad():
                    feat_dt, _ = teacher(drr, drr_head_n)
                feat_ds, pred_ds = model(drr, drr_head_n)
                loss_drr_cls = criterion(pred_ds.float(), targets)
                loss_drr_const = MSE(feat_ds.float(), feat_dt.float())
                loss_xdim = MSE(feat_s.float(), feat_dt.float()) + MSE(feat_ds.float(), feat_t.float())
                loss = loss + (1-coff) * loss_drr_cls + coff * (loss_drr_const + w_xdim * loss_xdim)

        (loss / accum_steps).backward()
        if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
            optimizer.step()
            optimizer.zero_grad()

        losses_cls.update(loss_cls.item(), samples1.size(0))
        losses_mse.update(loss_const.item(), samples1.size(0))
        if use_drr:
            losses_xdim.update(loss_xdim.item(), samples1.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            progress.display(i)
            save_image(_preview(samples1), "Models/student"+str(i))
            save_image(_preview(samples2), "Models/teacher"+str(i))

        if ema_mode == "iteration":
            ema_update_teacher(model, teacher, momentum_schedule, it)
            it += 1

    if ema_mode == "epoch":
        ema_update_teacher(model, teacher, momentum_schedule, it)
        it += 1

    if wandb.run is not None:
        wandb.log({"train_loss_cls_{}".format(dataset): losses_cls.avg})
        wandb.log({"train_loss_mse_{}".format(dataset): losses_mse.avg})


def train_one_epoch_interleaved(model, tasks, device, epoch, ema_mode, teacher, momentum_schedule, it,
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


def ema_update_teacher(model, teacher, momentum_schedule, it):
    with torch.no_grad():
        m = momentum_schedule[it]  # momentum parameter
        for param_q, param_k in zip(model.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


def evaluate(model, use_head_n, data_loader_val, device, criterion, dataset):
    model.eval()

    with torch.no_grad():
        batch_time = MetricLogger('Time', ':6.3f')
        losses = MetricLogger('Loss', ':.4e')
        progress = ProgressLogger(
        len(data_loader_val),
        [batch_time, losses], prefix='Val_'+dataset+': ')

        end = time.time()
        for i, (samples, _, targets) in enumerate(data_loader_val):
            samples, targets = samples.float().to(device), targets.float().to(device)

            _, outputs = model(samples, use_head_n)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), samples.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                progress.display(i)

    return losses.avg


def test_classification(model, use_head_n, data_loader_test, device, multiclass = False, is_3d = False): 
       
    model.eval()

    y_test = torch.FloatTensor().to(device)
    p_test = torch.FloatTensor().to(device)

    with torch.no_grad():
        for i, (samples, _, targets) in enumerate(tqdm(data_loader_test)):
            targets = targets.cuda()
            y_test = torch.cat((y_test, targets), 0)

            if is_3d:
                # (bs, 1, D, H, W) volumes: no TenCrop. Must not fall into the 5D TenCrop branch below.
                bs, n_crops = samples.size(0), 1
                varInput = samples.float().to(device)
            else:
                if len(samples.size()) == 4:
                    bs, c, h, w = samples.size()
                    n_crops = 1
                elif len(samples.size()) == 5:
                    bs, n_crops, c, h, w = samples.size()
                varInput = samples.view(-1, c, h, w).to(device)

            _, out = model(varInput, use_head_n)
            out = out.float()
            if multiclass:
                out = torch.softmax(out,dim = 1)
            else:
                out = torch.sigmoid(out)
            outMean = out.view(bs, n_crops, -1).mean(1)
            p_test = torch.cat((p_test, outMean.data), 0)

    return y_test, p_test
