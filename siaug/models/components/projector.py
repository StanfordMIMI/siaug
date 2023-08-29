import torch
from torch import nn

from siaug.models.components.blocks import linear_block

__all__ = ["Projector"]


class ProjectorConfig:
    def __init__(
        self,
        input_dim: int = 2048,
        output_dim: int = 2048,
        num_linear_layers: int = 3,
        batch_norm: bool = True,
        bias: bool = False,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_linear_layers = num_linear_layers
        self.batch_norm = batch_norm
        self.bias = bias


class Projector(nn.Module):
    """An MLP used to project the encoder's output into a lower dimension."""

    def __init__(self, cfg: ProjectorConfig) -> None:
        """
        Args:
            input_dim (int): output dim of the encoder
            output_dim (int): input dim of the predictor
            num_linear_layers (int, optional): number of linear layers including the final layer
            batch_norm (bool, optional): apply batch normalization between linear layers and ReLU
            bias (bool, optional): add a bias term to the linear layers
        """
        super().__init__()

        # store the config
        self.cfg = cfg

        # return the identity transform if we don't want a projector
        if cfg.num_linear_layers == 0:
            return nn.Identity()

        # construct the final linear_block
        layers = [nn.Linear(cfg.input_dim, cfg.output_dim, bias=cfg.bias)]
        if cfg.batch_norm:
            layers.append(nn.BatchNorm1d(cfg.output_dim, affine=False))

        # add num_linear_layers - 1 linear_blocks before the final linear_block
        for _ in range(cfg.num_linear_layers - 1):
            layers = linear_block(cfg.input_dim, cfg.input_dim, cfg.batch_norm, cfg.bias) + layers

        self.projector = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)
