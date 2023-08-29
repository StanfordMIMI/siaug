from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["ntxent_loss"]


def ntxent_loss(
    e1: Tensor,
    e2: Tensor,
    tau: float,
    weight: float = 1.0,
    num_stability: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Compute the NTXent loss.

    Args:
        e1 (Tensor): first set of embeddings.
        e2 (Tensor): second set of embeddings.
        tau (float): temperature of the loss.
        weight (float, optional): weight of the first loss term, the second is (1 - weight)
        num_stability (bool, optional): whether to use numerical stability.

    Returns:
        Tuple[Tensor, Tensor]: the loss.
    """
    e1, e2 = F.normalize(e1), F.normalize(e2)  # BxD, BxD
    logits = e1 @ e2.T / tau  # BxB

    # apply log-max trick for numerical stability
    if num_stability:
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

    labels = torch.arange(logits.size(0)).to(logits.device)  # B
    e1_loss = F.cross_entropy(logits, labels)  # 1
    e2_loss = F.cross_entropy(logits.T, labels)  # 1
    return weight * e1_loss - (1 - weight) * e2_loss
