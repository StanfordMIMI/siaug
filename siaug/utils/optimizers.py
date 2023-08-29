from apex.parallel.LARC import LARC
from torch import nn
from torch.optim import SGD

__all__ = ["LARS"]


def LARS(parameters: nn.Module, lr: float, **kwargs):
    """Wrap a SGD optimizer with LARS."""

    optimizer = SGD(parameters, lr=lr, **kwargs)
    return LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
