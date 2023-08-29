import os

import hydra
import pyrootutils
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from siaug.utils.extras import sanitize_dataloader_kwargs, set_seed
from siaug.utils.repr import repr_epoch
from siaug.utils.simsiam import CosineScheduler

# set the project root
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
config_dir = os.path.join(root, "configs")


@hydra.main(version_base="1.2", config_path=config_dir, config_name="train.yaml")
def main(cfg: DictConfig):
    """Train SimSiam based on a Hydra configuration (distributed)."""

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

    # create the model
    print(f"=> Creating model [device={device}]")
    model = cfg["model"].to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # optimizer
    # infer learning rate before changing the batch size
    print(f"=> Instantiating optimizer [device={device}]")
    params = model.get_param_groups() if hasattr(model, "get_param_groups") else model.parameters()
    init_lr = cfg["optimizer"].keywords["lr"] * cfg["dataloader"]["train"]["batch_size"] / 256
    optimizer = cfg["optimizer"](params, lr=init_lr)

    # learning rate scheduler
    print(f"=> Instantiating LR scheduler [device={device}]")
    scheduler = CosineScheduler(optimizer, cfg["max_epoch"])
    accelerator.register_for_checkpointing(scheduler)

    # move tensors to the correct device
    # NB: don't transform the scheduler, because it's called after each epoch
    train_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader,
        model,
        optimizer,
    )

    # resume from checkpoint
    start_epoch = cfg["start_epoch"]
    if cfg["resume_from_ckpt"] is not None:
        accelerator.load_state(cfg["resume_from_ckpt"])
        custom_ckpt = torch.load(os.path.join(cfg["resume_from_ckpt"], "custom_checkpoint_0.pkl"))
        start_epoch = custom_ckpt["last_epoch"]

    print(f"=> Starting model training [epochs={cfg['max_epoch']}]")
    for epoch in range(start_epoch, cfg["max_epoch"]):
        # train one epoch
        repr_epoch(
            epoch=epoch,
            accelerator=accelerator,
            dataloader=train_dataloader,
            model=model,
            criterion=cfg["criterion"],
            optimizer=optimizer,
            is_logging=is_logging,
            log_every_n_steps=cfg["log_every_n_steps"],
            fast_dev_run=cfg["fast_dev_run"],
        )

        # adjust the learning rate
        scheduler.step()

        # save checkpoint
        if (epoch + 1) % cfg["ckpt_every_n_epochs"] == 0:
            print(f"=> Saving checkpoint [epoch={epoch}]")
            accelerator.save_state(os.path.join(cfg["ckpt_dir"], f"epoch-{epoch:04d}.pt"))

    # save last model
    accelerator.save_state(os.path.join(cfg["ckpt_dir"], "last.pt"))

    print(f"=> Finished model training [epochs={cfg['max_epoch']}, device={device}]")
    accelerator.end_training()


if __name__ == "__main__":
    main()
