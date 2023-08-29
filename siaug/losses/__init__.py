import math
from typing import Dict

import torch
from torch import Tensor, nn

from .functional import ntxent_loss

__all__ = [
    "CosineSimilarityLoss",
    "SimSiamLoss",
    "NTXentLoss",
    "ConVIRTLoss",
    "TriPodLoss",
]


class CosineSimilarityLoss(nn.CosineSimilarity):
    """Mean-reduced negative cosine similarity loss."""

    def forward(self, x1, x2):
        return -1 * super().forward(x1, x2).mean()


class SimSiamLoss(nn.Module):
    """Loss function for SimSiam with support for weighted branches."""

    def __init__(
        self,
        criterion: nn.Module = CosineSimilarityLoss(),  # noqa
        weight: float = 0.5,
        stopgrad: bool = True,
    ):
        super().__init__()
        self.criterion, self.weight, self.stopgrad = criterion, weight, stopgrad

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        # stopgrad z1 and z2
        p1, p2, z1, z2 = inputs["p1"], inputs["p2"], inputs["z1"], inputs["z2"]
        if self.stopgrad:
            z1, z2 = inputs["z1"].detach(), inputs["z2"].detach()

        return self.weight * self.criterion(p1, z2) + (1 - self.weight) * self.criterion(p2, z1)


class NTXentLoss(nn.Module):
    """NTXent loss function.

    See ntxent_loss for more details.
    """

    def __init__(self, tau: float, weight: float = 1.0, num_stability: bool = False):
        super().__init__()
        self.tau = tau
        self.weight = weight
        self.num_stability = num_stability

    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> Tensor:
        return ntxent_loss(e1, e2, self.tau, self.weight, self.num_stability)


class ConVIRTLoss(nn.Module):
    """ConVIRT loss function.

    This is a special variant of the InfoNCE loss function, also known as the
    NTXentLoss or the Contrastive Loss.

    Args:
        tau (float): temperature of the loss.
        weight (float, optional): weight of the first loss term, the second is (1 - weight).
        num_stability (bool, optional): whether to use numerical stability.
    """

    def __init__(self, tau: float = 0.1, weight: float = 0.75, num_stability: bool = False):
        super().__init__()

        if weight < 0 or weight > 1:
            raise ValueError("The weight must be between 0 and 1.")

        self.tau, self.weight, self.num_stability = tau, weight, num_stability

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        img_proj, txt_proj = inputs["z1"], inputs["z2"]
        return ntxent_loss(img_proj, txt_proj, self.tau, self.weight, self.num_stability)


class TriPodLoss(nn.Module):
    """Generic loss function for TriPod.

    The TriPodLoss defaults to a weighted combination of CosineSimilarityLosses
    between the different projectors and predictors.
    NB: the lambdas can be used to disregard one or more branches/connections.

    SimSiam:
    The TriPodLoss is able to mimic the SimSiam loss function by setting the
    lambdas to 1/2 for p1_z2 and p2_z1.

    ConVIRT:
    The TriPodLoss is able to mimic the ConVIRT loss function by setting the
    lambdas to 1/2 for p3_z5 and p5_z4, as long as the predictors of these two
    branches are Identity() layers.

    TriPod Contrastive:
    The contrastive loss between a predictor and a projection head can be computed
    using the NTXentLoss criterion with a weight of 1.0.

    Args:
        criterion (nn.Module, optional): criterion to use for the loss computation.
        lambdas (Dict[str, float], optional): lambdas for the different loss terms.
        stopgrad (bool, optional): whether to stop gradients for the projections.
    """

    def __init__(
        self,
        criterion: nn.Module = CosineSimilarityLoss(),  # noqa: B008
        lambdas: Dict[str, float] = {  # noqa: B006
            "p1_z2": 1 / 12,  # i2i <-> i2i (img1)
            "p1_z6": 1 / 12,  # i2i <-> t2i (img1)
            "p2_z1": 1 / 12,  # i2i <-> i2i (img2)
            "p2_z6": 1 / 12,  # i2i <-> t2i (img2)
            "p3_z4": 1 / 12,  # i2t <-> i2t (img1)
            "p3_z5": 1 / 12,  # i2t <-> t2t (img1)
            "p4_z3": 1 / 12,  # i2t <-> i2t (img2)
            "p4_z5": 1 / 12,  # i2t <-> t2t (img2)
            "p5_z3": 1 / 12,  # t2t <-> i2t (img1)
            "p5_z4": 1 / 12,  # t2t <-> i2t (img2)
            "p6_z1": 1 / 12,  # t2i <-> i2i (img1)
            "p6_z2": 1 / 12,  # t2i <-> i2i (img2)
        },
        stopgrad: bool = True,
    ):
        super().__init__()

        assert math.isclose(
            sum(lambdas.values()), 1.0
        ), "Sum of the lambas for the TriPodLoss needs to be 1."

        self.criterion = criterion
        self.lambdas = lambdas
        self.stopgrad = stopgrad

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        if self.stopgrad:
            for k, v in inputs.items():
                if k.startswith("z"):
                    inputs[k] = v.detach()

        loss = 0
        for k, v in self.lambdas.items():
            pred, _, proj = k.partition("_")
            loss += v * self.criterion(inputs[pred], inputs[proj])

        return loss
