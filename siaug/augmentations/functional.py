from numbers import Number
from typing import Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor

__all__ = ["identity", "sobel"]


def identity(x):
    return x


def _format_image(arr) -> torch.Tensor:
    """Converts an image into a tensor with a channel dimension."""

    if not isinstance(arr, torch.Tensor):
        arr = to_tensor(arr)

    ndim = arr.ndim
    if ndim == 4:
        arr = arr.squeeze(0)
    elif ndim == 2:
        arr = arr.unsqueeze(0)

    return arr


def sobel(
    arr: torch.Tensor,
    pool: bool = True,
    clamp: Tuple[Number, Number] = (0, 255),
) -> torch.Tensor:
    """Convolve a tensor using the sobel kernels.

    Examples:
    >>> # apply sobel filter
    >>> img = Image.open("/data3/sluijs/raw/chexpert/train/patient26265/study1/view1_frontal.jpg")
    >>> arr = torch.tensor(np.asarray(img), dtype=torch.float).unsqueeze(0).unsqueeze(0)
    """

    arr = _format_image(arr)
    num_channels = arr.shape[0]
    kernel_h = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float)
    kernel_v = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float)
    kernels = torch.stack([kernel_h, kernel_v]).unsqueeze(1).expand(2, num_channels, 3, 3)
    out = F.conv2d(arr.unsqueeze(0), kernels, padding=1)

    if not pool:
        return out

    # pool, clamp, and format the output
    out = torch.sqrt(torch.pow(out[:, 0, ...], 2) + torch.pow(out[:, 1, ...], 2) + 1e-6)
    if clamp:
        minv, maxv = clamp
        out = torch.clamp(out, minv, maxv)

    if num_channels > 1:
        out = out.repeat(num_channels, 1, 1)

    return out
