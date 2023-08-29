from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from kornia.augmentation import AugmentationBase2D, IntensityAugmentationBase2D
from kornia.filters import sobel
from torch import Tensor

from siaug.augmentations import functional, text
from siaug.augmentations.functional import *  # noqa
from siaug.augmentations.text import *  # noqa

__all__ = [
    "ToSiamese",
    "ExtractKeys",
    "RandomSobelFilter",
    "RandomGaussianNoise",
    "Clamp",
    "RandomRotate90",
    "ExpandChannels",
]
__all__.extend(text.__all__)
__all__.extend(functional.__all__)


class ToSiamese:
    """Split augmentation pipeline into two branches."""

    def __init__(self, t1: Callable, t2: Callable):
        self.t1, self.t2 = t1, t2

    def __call__(self, x: Any) -> Tuple[Any, Any]:
        return [self.t1(x), self.t2(x)]


class ExtractKeys:
    """Extract keys from a Dict sample."""

    def __init__(self, keys: List[str], unpack: bool = False):
        self.keys, self.unpack = keys, unpack

    def __call__(self, sample: Dict) -> Union[Dict, Any]:
        sample = {k: v for (k, v) in sample.items() if k in self.keys}

        if not self.unpack:
            return sample

        keys = list(sample.keys())
        if len(keys) > 1:
            raise ValueError("Can't unpack dicts with more than 1 key.")

        return sample[keys[0]]


class RandomSobelFilter(IntensityAugmentationBase2D):
    """Apply a sobel filter."""

    def apply_transform(self, input: Tensor, **kwargs) -> Tensor:
        return sobel(input)


class RandomGaussianNoise(IntensityAugmentationBase2D):
    r"""Add gaussian noise to a batch of multi-dimensional images.

    .. image:: _static/img/RandomGaussianNoise.png

    Args:
        mean: The mean of the gaussian distribution.
        std: The standard deviation or range of standard deviations of the gaussian distribution.
        same_on_batch: apply the same transformation across the batch.
        p: probability of applying the transformation.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).
    Examples:
        >>> rng = torch.manual_seed(0)
        >>> img = torch.ones(1, 1, 2, 2)
        >>> RandomGaussianNoise(mean=0., std=1., p=1.)(img)
        tensor([[[[ 2.5410,  0.7066],
                  [-1.1788,  1.5684]]]])

    To apply the exact augmenation again, you may take the advantage of the
    previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomGaussianNoise(mean=0., std=1., p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: Union[Number, Tuple[Number, Number]] = 1.0,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
        return_transform: Optional[bool] = None,
    ) -> None:
        super().__init__(
            p=p,
            return_transform=return_transform,
            same_on_batch=same_on_batch,
            p_batch=1.0,
            keepdim=keepdim,
        )

        self.flags = dict(mean=mean, std=std)

    def generate_parameters(self, shape: torch.Size) -> Dict[str, Tensor]:
        std = torch.rand(1)
        noise = torch.randn(shape)
        return dict(noise=noise, std=std)

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], **kwargs
    ) -> Tensor:
        std = flags["std"]
        if not isinstance(std, Number):
            lb, ub = std
            std = (ub - lb) * params["std"] + lb

        return input + params["noise"].to(input.device) * std + flags["mean"]


class Clamp:
    """Clamp and optionally scale tensor to 0-1."""

    def __init__(self, lb: Number, ub: Number, scale: bool = False, inplace: bool = False):
        self.flags = dict(lb=lb, ub=ub, scale=scale)

    def __call__(self, input: Tensor):
        lb, ub = self.flags["lb"], self.flags["ub"]
        if self.flags["inplace"]:
            out = torch.clamp_(input, lb, ub)
            if self.flags["scale"]:
                out = out.sub_(lb).div_(ub - lb)
        else:
            out = torch.clamp(input, lb, ub)
            if self.flags["scale"]:
                out = (out - lb) / (ub - lb)

        return out


class RandomRotate90(AugmentationBase2D):
    def generate_parameters(self, shape: torch.Size) -> Dict[str, Tensor]:
        k = int(torch.randint(0, 3, (1,)))
        return dict(k=k)

    def apply_transform(self, input: Tensor, params: Dict[str, Tensor], **kwargs) -> Tensor:
        return torch.rot90(input, k=params["k"], dims=(-2, -1))

    def compute_transformation(self, input: Tensor, **kwargs) -> Tensor:
        return self.identity_matrix(input)


class ExpandChannels:
    def __init__(self, out_channels: int = 3):
        self.out_channels = out_channels

    def __call__(self, x: torch.Tensor):
        if x.ndim != 3 or x.shape[0] != 1:
            raise NotImplementedError

        return x.expand(self.out_channels, -1, -1)
