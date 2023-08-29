import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter


class ToSimSiam:
    def __init__(self, normalize: bool = True):
        transforms = [
            T.RandomResizedCrop(224, scale=(0.2, 1)),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23)], p=0.5),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]

        if normalize:
            transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        self.transform = T.Compose(transforms)

    def __call__(self, x: Image) -> torch.Tensor:
        return self.transform(x)


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
