from typing import Dict, Tuple, Union

import timm
from torch import Tensor, nn

from siaug.models.components import (
    Predictor,
    PredictorConfig,
    Projector,
    ProjectorConfig,
    TextEncoder,
)

__all__ = ["TriPod"]


class TriPod(nn.Module):
    """TriPod model.

    This model is a hybrid between img-to-img and img-to-text SSL architectures. It consists of two
    identical image branches, that share weights, and a text branch.

    Examples:
    >>> model = Tripod(...)
    >>> tokenize = Tokenizer(...)
    >>> x1 = torch.rand((5, 3, 224, 224))
    >>> x2 = torch.rand((5, 3, 224, 224))
    >>> x3 = tokenize(["text1", "text2", "text3", "text", "text5"])
    >>> model(x1, x2, x3)
    """

    def __init__(
        self,
        img_backbone: str,
        num_channels: int,
        txt_backbone: str,
        projectors: Dict[str, ProjectorConfig],
        predictors: Dict[str, PredictorConfig],
        embedding_method: str = "last_hidden_state_cls",
        **kwargs,
    ):
        super().__init__()

        # encoders
        kwargs = {**kwargs, "model_name": img_backbone, "in_chans": num_channels, "num_classes": 0}
        self.img_encoder: nn.Module = timm.create_model(**kwargs)
        self.txt_encoder = TextEncoder(txt_backbone, embedding_method)

        # projectors
        # NB: do not automagically change the projector's input_dim
        for k, shared in [("i2i", "img"), ("i2t", "img"), ("t2t", "txt"), ("t2i", "txt")]:
            cfg = projectors.get(k, projectors[shared])
            setattr(self, f"{k}_projector", Projector(cfg))

        # predictors
        for k in ["img", "txt"]:
            cfg = predictors.get(k, predictors["all"])
            setattr(self, f"{k}_predictor", Predictor(cfg))

    def forward(
        self,
        inputs: Dict[str, Union[Tuple[Tensor, Tensor], Dict[str, Tensor]]],
    ) -> Tuple[Tensor, ...]:
        (x1, x2), x3 = inputs["img"], inputs["txt"]

        # encode branches
        h1 = self.img_encoder(x1)
        h2 = self.img_encoder(x2)
        h3 = self.txt_encoder(x3)

        # compute the projected features between each of the branches = 2 * 3
        z1 = self.i2i_projector(h1)
        z2 = self.i2i_projector(h2)
        z3 = self.i2t_projector(h1)
        z4 = self.i2t_projector(h2)
        z5 = self.t2t_projector(h3)
        z6 = self.t2i_projector(h3)

        # compute the output of the predictors for the images
        p1 = self.img_predictor(z1)
        p2 = self.img_predictor(z2)
        p3 = self.txt_predictor(z3)
        p4 = self.txt_predictor(z4)
        p5 = self.txt_predictor(z5)
        p6 = self.img_predictor(z6)

        # return the predictions and detach the projected features
        return {
            "p1": p1,
            "p2": p2,
            "p3": p3,
            "p4": p4,
            "p5": p5,
            "p6": p6,
            "z1": z1,
            "z2": z2,
            "z3": z3,
            "z4": z4,
            "z5": z5,
            "z6": z6,
        }

    def get_param_groups(self, recurse: bool = True):
        # convert into param_groups with annotations on whether the lr needs to be fixed
        return [
            {
                "name": "img_encoder",
                "params": self.img_encoder.parameters(recurse),
                "fixed_lr": False,
            },
            {
                "name": "txt_encoder",
                "params": self.txt_encoder.parameters(recurse),
                "fixed_lr": False,
            },
            {
                "name": "i2i_projector",
                "params": self.i2i_projector.parameters(recurse),
                "fixed_lr": False,
            },
            {
                "name": "i2t_projector",
                "params": self.i2t_projector.parameters(recurse),
                "fixed_lr": False,
            },
            {
                "name": "t2i_projector",
                "params": self.t2i_projector.parameters(recurse),
                "fixed_lr": False,
            },
            {
                "name": "t2t_projector",
                "params": self.t2t_projector.parameters(recurse),
                "fixed_lr": False,
            },
            {
                "name": "img_predictor",
                "params": self.img_predictor.parameters(recurse),
                "fixed_lr": True,
            },
            {
                "name": "txt_predictor",
                "params": self.txt_predictor.parameters(recurse),
                "fixed_lr": True,
            },
        ]
