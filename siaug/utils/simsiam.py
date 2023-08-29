import math

from torch.optim.lr_scheduler import _LRScheduler


class CosineScheduler(_LRScheduler):
    def __init__(self, optimizer, max_epochs: int, verbose: bool = False):
        self.optimizer = optimizer
        self.max_epochs = max_epochs

        # last epoch should always be -1, use load_state_dict to resume
        super().__init__(optimizer, -1, verbose)

    def _compute_lr(self, param_group):
        init_lr = param_group["initial_lr"]
        current_lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * self.last_epoch / self.max_epochs))

        if "fixed_lr" in param_group and param_group["fixed_lr"]:
            return init_lr
        else:
            return current_lr

    def get_lr(self):
        return [self._compute_lr(param_group) for param_group in self.optimizer.param_groups]


# TODO: replace with torchmetrics
class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
