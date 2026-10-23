"""
lr_schedule.py  (place next to engine.py)

Complete replacement for timm's create_scheduler in Ark+ pretraining. The learning rate is a pure
function of the epoch index and is set at the START of every epoch:

    warm-up (epoch < W):  lr = start + (peak - start) * epoch / (W - 1)                 -> exactly `peak` at epoch W-1
    cosine  (epoch >= W): lr = end + 0.5 * (peak - end) * (1 + cos(pi * (epoch - W + 1) / (T - W)))
                                                                                            -> exactly `end` at epoch T-1
    T = --pretrain_epochs, W = --warmup-epochs, start = --warmup-lr, peak = --lr, end = --min-lr

Same formula as the WarmupCosineScheduler in the team's revised engine.py, so runs are comparable.
No dependence on the validation loss, on --epochs or on --cooldown-epochs. Because the lr depends only
on the epoch, resuming needs no saved scheduler state (old timm scheduler states are accepted and ignored).
"""
import math


class WarmupCosineScheduler:
    def __init__(self, optimizer, total_epochs, warmup_epochs, start_lr, peak_lr, end_lr):
        if total_epochs < 3:
            raise ValueError("need at least 3 epochs")
        if warmup_epochs < 2 or warmup_epochs >= total_epochs:
            raise ValueError("--warmup-epochs must be >= 2 and < --pretrain_epochs")
        if start_lr > peak_lr or end_lr > peak_lr:
            raise ValueError("--warmup-lr and --min-lr must not exceed --lr")
        self.optimizer = optimizer
        self.T, self.W = int(total_epochs), int(warmup_epochs)
        self.start, self.peak, self.end = float(start_lr), float(peak_lr), float(end_lr)
        self.last_epoch = -1

    def lr_at(self, epoch):
        if epoch < 0 or epoch >= self.T:
            raise ValueError("epoch {} outside [0, {})".format(epoch, self.T))
        if epoch < self.W:
            return self.start + (self.peak - self.start) * epoch / float(self.W - 1)
        progress = (epoch - self.W + 1) / float(self.T - self.W)
        return self.end + 0.5 * (self.peak - self.end) * (1.0 + math.cos(math.pi * progress))

    def step(self, epoch):
        lr = self.lr_at(epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_epoch = epoch
        return lr

    def describe(self, every=10):
        rows = ["epoch {:4d}  lr {:.6g}".format(e, self.lr_at(e)) for e in range(self.T)
                if e % every == 0 or e in (self.W - 1, self.W, self.T - 1)]
        return "lr schedule (warmup-cosine, T={}, W={}, start={}, peak={}, end={}):\n  ".format(
            self.T, self.W, self.start, self.peak, self.end) + "\n  ".join(rows)

    def state_dict(self):
        return {"kind": "warmup_cosine", "last_epoch": self.last_epoch, "T": self.T, "W": self.W,
                "start": self.start, "peak": self.peak, "end": self.end}

    def load_state_dict(self, state):
        # the lr is recomputed from the epoch at every step, so nothing needs restoring;
        # refuse only a checkpoint from this scheduler with different settings
        if isinstance(state, dict) and state.get("kind") == "warmup_cosine":
            for k in ("T", "W", "start", "peak", "end"):
                if state.get(k) != getattr(self, k):
                    raise ValueError("checkpoint lr schedule differs in {}: {} vs {}".format(k, state.get(k), getattr(self, k)))
            self.last_epoch = state.get("last_epoch", -1)
