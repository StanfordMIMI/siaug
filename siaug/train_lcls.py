import os

import hydra
import pyrootutils
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from siaug.utils.eval import lcls_eval
from siaug.utils.extras import sanitize_dataloader_kwargs, set_seed
from siaug.utils.lcls import lcls_epoch
from siaug.utils.simsiam import CosineScheduler

# set the project root
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
config_dir = os.path.join(root, "configs")


@hydra.main(version_base="1.2", config_path=config_dir, config_name="train.yaml")
def main(cfg: DictConfig):
    """Linearly evaluate a pretrained representation based on a Hydra configuration."""

    print(f"=> Starting [experiment={cfg['task_name']}]")
    print("=> Initializing Hydra configuration")
    cfg = instantiate(cfg)

    seed = cfg.get("seed", None)
    if seed is not None:
        set_seed(seed)

    # setup accelerator
    is_logging = cfg.get("logger", None) is not None
    print(f"=> Instantiate accelerator [logging={is_logging}]")
    logger_name = "wandb" if is_logging else None
    logger_kwargs = {"wandb": cfg.get("logger", None)}

    accelerator = Accelerator(log_with=logger_name, split_batches=True)
    accelerator.init_trackers("siaug", config=cfg, init_kwargs=logger_kwargs)
    device = accelerator.device

    # instantiate dataloaders
    print(f"=> Instantiating train dataloader [device={device}]")
    train_dataloader = DataLoader(**sanitize_dataloader_kwargs(cfg["dataloader"]["train"]))

    print(f"=> Instantiating valid dataloader [device={device}]")
    valid_dataloader = DataLoader(**sanitize_dataloader_kwargs(cfg["dataloader"]["valid"]))

    # create the model
    print(f"=> Creating model [device={device}]")
    model = cfg["model"].to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # build optimizer from a partial optimizer
    print(f"=> Instantiating the optimizer [device={device}")
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    init_lr = cfg["optimizer"].keywords["lr"] * cfg["dataloader"]["train"]["batch_size"] / 256
    optimizer = cfg["optimizer"](params, lr=init_lr)

    # learning rate scheduler
    print(f"=> Instantiating LR scheduler [device={device}]")
    scheduler = CosineScheduler(optimizer, cfg["max_epoch"])
    accelerator.register_for_checkpointing(scheduler)

    # prepare the components for multi-gpu/mixed precision training
    train_dataloader, valid_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader,
        valid_dataloader,
        model,
        optimizer,
    )

    # resume from checkpoint
    start_epoch = cfg["start_epoch"]
    if cfg["resume_from_ckpt"] is not None:
        accelerator.load_state(cfg["resume_from_ckpt"])
        custom_ckpt = torch.load(os.path.join(cfg["resume_from_ckpt"], "custom_checkpoint_0.pkl"))
        start_epoch = custom_ckpt["last_epoch"]

    # setup metrics
    max_metric = None

    print(f"=> Starting model training [epochs={cfg['max_epoch']}]")
    for epoch in range(start_epoch, cfg["max_epoch"]):
        # train one epoch
        lcls_epoch(
            epoch=epoch,
            accelerator=accelerator,
            dataloader=train_dataloader,
            model=model,
            criterion=cfg["criterion"],
            optimizer=optimizer,
            metrics=cfg["metrics"],
            is_logging=is_logging,
            log_every_n_steps=cfg["log_every_n_steps"],
            fast_dev_run=cfg["fast_dev_run"],
        )

        # adjust the learning rate per epoch
        scheduler.step()

        # evaluate the model
        metric = lcls_eval(
            accelerator=accelerator,
            dataloader=valid_dataloader,
            model=model,
            criterion=cfg["criterion"],
            metrics=cfg["metrics"],
            is_logging=is_logging,
            log_every_n_steps=cfg["log_every_n_steps"],
            fast_dev_run=cfg["fast_dev_run"],
        )

        # save the best model
        if max_metric is None or metric > max_metric:
            accelerator.save_state(os.path.join(cfg["ckpt_dir"], "best.pt"))
            max_metric = metric

        # save checkpoint
        if (epoch + 1) % cfg["ckpt_every_n_epochs"] == 0:
            print(f"=> Saving checkpoint [epoch={epoch}]")
            accelerator.save_state(os.path.join(cfg["ckpt_dir"], f"epoch-{epoch:04d}.pt"))

    # save last model
    accelerator.save_state(os.path.join(cfg["ckpt_dir"], "last.pt"))

    print(f"=> Finished model training [epochs={cfg['max_epoch']}, metric={max_metric}]")
    accelerator.end_training()


if __name__ == "__main__":
    main()
