from typing import Dict, Tuple

import timm
from torch import Tensor, nn

from siaug.models.components import (
    Predictor,
    PredictorConfig,
    Projector,
    ProjectorConfig,
)

__all__ = ["SimSiam"]


class SimSiam(nn.Module):
    """Implementation of the SimSiam model.

    URL: https://arxiv.org/pdf/2011.10566.pdf
    https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py
    """

    def __init__(
        self,
        backbone: str,
        num_channels: int,
        output_dim: int = 2048,
        pred_hidden_dim: int = 512,
        **kwargs,
    ):

        super().__init__()

        # encoder network
        kwargs = {
            **kwargs,
            "model_name": backbone,
            "in_chans": num_channels,
            "num_classes": 0,
        }
        self.encoder: nn.Module = timm.create_model(**kwargs)

        # projection (mlp) network
        self.projector = Projector(
            ProjectorConfig(
                input_dim=self.encoder.num_features,
                output_dim=output_dim,
            )
        )

        # predictor network (h)
        self.predictor = Predictor(
            PredictorConfig(
                dim=output_dim,
                hidden_dim=pred_hidden_dim,
            )
        )

    def forward(self, inputs: Dict[str, Tuple[Tensor, Tensor]]) -> Tuple[Tensor, ...]:
        x1, x2 = inputs["img"]

        # compute features for each view
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))

        # compute the output of the predictors
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return {"p1": p1, "p2": p2, "z1": z1, "z2": z2}

    def get_param_groups(self, recurse: bool = True):
        # convert into param_groups with annotations on whether the lr needs to be fixed
        return [
            {
                "name": "encoder",
                "params": self.encoder.parameters(recurse),
                "fixed_lr": False,
            },
            {
                "name": "projector",
                "params": self.projector.parameters(recurse),
                "fixed_lr": False,
            },
            {
                "name": "predictor",
                "params": self.predictor.parameters(recurse),
                "fixed_lr": True,
            },
        ]
