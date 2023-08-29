from typing import List

from torch import nn

__all__ = ["linear_block"]


def linear_block(
    input_dim: int,
    output_dim: int,
    batch_norm: bool = True,
    bias: bool = False,
) -> List[nn.Module]:
    """Construct a non-final projector/predictor linear_block.

    Args:
        input_dim (int): number of input features
        output_dim (int): number of output features
        batch_norm (bool, optional): apply batch norm between linear layer and ReLU
        bias (bool, optional): add a bias term to the linear layers
    """
    layers = [nn.Linear(input_dim, output_dim, bias=False), nn.ReLU(inplace=True)]
    if batch_norm:
        layers.insert(1, nn.BatchNorm1d(output_dim))

    return layers
