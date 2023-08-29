from time import time
from typing import Callable, Dict

from accelerate import Accelerator
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from siaug.utils.simsiam import AverageMeter, ProgressMeter

__all__ = ["repr_epoch"]


def repr_epoch(
    epoch: int,
    accelerator: Accelerator,
    dataloader: DataLoader,
    model: nn.Module,
    criterion: Callable[[Dict[str, Tensor]], Tensor],
    optimizer: Optimizer,
    is_logging: bool = False,
    log_every_n_steps: int = 1,
    fast_dev_run: bool = False,
):
    """Pretrain a single epoch."""

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()

    end = time()
    for i, batch in enumerate(dataloader):
        # measure data loading time
        data_time.update(time() - end)

        # compute output and loss
        batch_size = batch["img"][0].size(0)
        output = model(batch)

        loss = criterion(output)
        losses.update(loss.item(), batch_size)

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

            # registered loggers
            if is_logging:
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

                for idx, pg in enumerate(optimizer.param_groups):
                    name = pg["name"] if "name" in pg else f"param_group_{idx}"
                    log_data[f"{name}/momentum"] = pg.get("momentum", 0)
                    log_data[f"{name}/weight_decay"] = pg.get("weight_decay", 0)
                    log_data[f"{name}/lr"] = pg.get("lr", 0)

                accelerator.log(log_data)

        if fast_dev_run:
            break
