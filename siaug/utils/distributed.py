import torch.distributed as dist


def setup(rank: int, world_size: int, enabled: bool = False):
    """Initialize the process group and set the master server."""

    if enabled:
        assert world_size > 1, "WORLD_SIZE should be larger than 1."

        dist.init_process_group("nccl", rank=rank, world_size=world_size)


def is_distributed():
    """Check whether this is a distributed run."""

    return dist.is_initialized()


def terminate():
    """Terminate the process group."""

    if is_distributed():
        dist.destroy_process_group()


def is_master() -> bool:
    """Check whether this process is the master process."""

    # consider a single process application to be the master process
    if not is_distributed():
        return True

    return dist.get_rank() == 0
