from time import time

import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from siaug.utils.simsiam import AverageMeter, ProgressMeter

__all__ = ["convirt_eval"]


def convirt_eval(
    accelerator: Accelerator,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    is_logging: bool = False,
    log_every_n_steps: int = 1,
    fast_dev_run: bool = False,
):
    """Validate a single epoch."""

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")

    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses],
        prefix="Test: ",
    )

    # switch to eval mode (train mode might change BatchNorm running mean/std)
    model.eval()

    with torch.no_grad():
        end = time()
        for i, batch in enumerate(dataloader):
            # measure data loading time
            data_time.update(time() - end)

            # compute output and loss
            batch_size = batch["img"][0].size(0)
            output = model(batch)

            loss = criterion(output)
            losses.update(loss.item(), batch_size)

            # measure elapsed time
            batch_time.update(time() - end)
            end = time()

            # print results
            if i % log_every_n_steps == 0:
                progress.display(i)

                # wandb
                if is_logging:
                    # log progress, loss, and performance
                    log_data = {
                        "valid/epoch_loss": losses.avg,
                        "valid/epoch_batch_time": batch_time.avg,
                        "valid/epoch_data_time": data_time.avg,
                        "valid/step_loss": losses.val,
                        "valid/step_data_time": data_time.val,
                        "valid/step_batch_time": batch_time.val,
                    }

                    accelerator.log(log_data)

            if fast_dev_run:
                break

    return losses.avg
