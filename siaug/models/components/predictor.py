import torch
from torch import nn

from siaug.models.components.blocks import linear_block

__all__ = ["Predictor"]


class PredictorConfig:
    def __init__(
        self,
        dim: int = 2048,
        hidden_dim: int = 2048,
        num_linear_layers: int = 2,
        batch_norm: bool = True,
        bias: bool = True,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_linear_layers = num_linear_layers
        self.batch_norm = batch_norm
        self.bias = bias


class Predictor(nn.Module):
    """A predictor that must be used on top of a projector to predict the representation generated
    by the teacher.

    Args:
        dim (int, optional): input and output dimension of the predictor
        hidden_dim (int, optional): number of features in the intermediate layers
        batch_norm (bool, optional): apply batch norm between linear layers and ReLU
        bias (bool, optional): add a bias term to the linear layers
    """

    def __init__(self, cfg: PredictorConfig) -> None:
        super().__init__()

        # store the config
        self.cfg = cfg

        # return identity to mimic predictor-less networks
        if cfg.num_linear_layers == 0:
            return nn.Identity()

        # construct the final layer
        hidden_dim = cfg.dim if cfg.num_linear_layers == 1 else cfg.hidden_dim
        layers = [nn.Linear(hidden_dim, cfg.dim, bias=cfg.bias)]

        # construct the blocks to prepend
        for _ in range(cfg.num_linear_layers - 1):
            layers = linear_block(cfg.dim, hidden_dim, cfg.batch_norm, cfg.bias) + layers

        self.predictor = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)
