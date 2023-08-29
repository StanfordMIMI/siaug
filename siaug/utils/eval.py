from time import time
from typing import Callable, List, Tuple

import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from siaug.utils.simsiam import AverageMeter, ProgressMeter

__all__ = ["lcls_eval"]


def lcls_eval(
    accelerator: Accelerator,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    metrics: List[Tuple[str, Callable[[torch.Tensor, torch.Tensor], float]]] = [],
    is_logging: bool = False,
    log_every_n_steps: int = 1,
    fast_dev_run: bool = False,
):
    """Validate a single epoch."""

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    _metrics = [AverageMeter(name) for (name, _) in metrics]

    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses] + _metrics,
        prefix="Test: ",
    )

    # switch to eval mode (train mode might change BatchNorm running mean/std)
    model.eval()

    with torch.no_grad():
        end = time()
        for i, batch in enumerate(dataloader):
            images, targets = batch["img"], batch["lbl"]

            # measure data loading time
            data_time.update(time() - end)

            output = model(images)
            # BCE loss requires float targets
            loss = criterion(output, targets.float())

            # metrics
            losses.update(loss.item(), images.size(0))

            output, targets = accelerator.gather_for_metrics((output, targets))
            for idx, (_, metric) in enumerate(metrics):
                _metrics[idx].update(metric(output, targets).item(), images.size(0))

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

                    # log metrics
                    for idx, (name, _) in enumerate(metrics):
                        log_data[f"valid/epoch_{name}"] = _metrics[idx].avg
                        log_data[f"valid/step_{name}"] = _metrics[idx].val

                    accelerator.log(log_data)

            if fast_dev_run:
                break

    out_metric = _metrics[0].avg if len(_metrics) > 0 else losses.avg
    return out_metric
