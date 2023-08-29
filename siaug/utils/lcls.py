from time import time
from typing import Callable, List, Tuple

import torch
from accelerate import Accelerator
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from siaug.utils.simsiam import AverageMeter, ProgressMeter

__all__ = ["lcls_epoch"]


def lcls_epoch(
    epoch: int,
    accelerator: Accelerator,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    metrics: List[Tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]] = [],
    is_logging: bool = False,
    log_every_n_steps: int = 1,
    fast_dev_run: bool = False,
):
    """Train one epoch for linear classification."""

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    _metrics = [AverageMeter(name) for (name, _) in metrics]

    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to eval mode (train mode might change BatchNorm running mean/std)
    model.eval()

    end = time()
    for i, batch in enumerate(dataloader):
        images, targets = batch["img"], batch["lbl"]

        # measure data loading time
        data_time.update(time() - end)

        # compute output and loss
        output = model(images)

        # BCE loss requires float targets
        loss = criterion(output, targets.float())
        losses.update(loss.item(), images.size(0))

        # update metrics
        for idx, (_, metric) in enumerate(metrics):
            _metrics[idx].update(metric(output, targets).item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        # gradient scaling for mixed precision
        accelerator.backward(loss)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time() - end)
        end = time()

        # step logging
        if i % log_every_n_steps == 0:
            # stdout
            progress.display(i)

            # wandb
            if is_logging:
                # log progress, loss, and performance
                log_data = {
                    "epoch": epoch,
                    "epoch/loss": losses.avg,
                    "epoch/batch_time": batch_time.avg,
                    "epoch/data_time": data_time.avg,
                    "step": i,
                    "step/global": len(dataloader) * epoch + i,
                    "step/loss": losses.val,
                    "step/data_time": data_time.val,
                    "step/batch_time": batch_time.val,
                }

                # log metrics
                for idx, (name, _) in enumerate(metrics):
                    log_data[f"epoch/{name}"] = _metrics[idx].avg
                    log_data[f"step/{name}"] = _metrics[idx].val

                for idx, pg in enumerate(optimizer.param_groups):
                    name = pg["name"] if "name" in pg else f"param_group_{idx}"
                    # log_data[f"{name}/momentum"] = pg["momentum"]
                    log_data[f"{name}/weight_decay"] = pg["weight_decay"]
                    log_data[f"{name}/lr"] = pg["lr"]

                accelerator.log(log_data)

        if fast_dev_run:
            break
