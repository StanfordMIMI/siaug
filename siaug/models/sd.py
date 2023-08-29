import torch
from diffusers import AutoencoderKL
from torch import nn

__all__ = ["SDEncoder"]


class SDEncoder(nn.Module):
    def __init__(
        self, num_classes: int = 8, pretrained: bool = False, freeze: bool = False, **kwargs
    ):
        super().__init__()

        if not pretrained:
            raise ValueError("SDEncoder requires pretrained=True")

        vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="vae",
        )

        self.encoder = vae.encoder
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(8 * 64 * 64, num_classes)

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)  # Bx8x64x64
        x = self.flatten(x)  # Bx32_768
        return self.linear(x)  # num_classes
