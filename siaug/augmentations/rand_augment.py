import kornia.augmentation as ka
import torch
import torchvision.transforms as tt

from siaug.augmentations import RandomGaussianNoise, RandomRotate90, RandomSobelFilter


class RandAugment:
    def __init__(self, n: int = 2, rrc: bool = False, jitter: bool = False):
        self.n = n
        self.rrc = rrc
        self.jitter = jitter

    def __call__(self, img):
        # Define a list of possible augmentations
        augmentations = [
            # Kornia
            RandomRotate90(keepdim=True, p=1),
            RandomGaussianNoise(std=(0.01, 0.03), keepdim=True, p=1),
            RandomSobelFilter(keepdim=True, p=1),
            ka.RandomGaussianBlur((23, 23), sigma=(0.1, 2.0), keepdim=True, p=1),
            ka.RandomThinPlateSpline(keepdim=True, p=1),
            ka.RandomErasing(keepdim=True, p=1),
            ka.RandomMotionBlur((23, 23), angle=(0, 360), direction=(0, 1), keepdim=True, p=1),
            ka.RandomJigsaw(grid=(4, 4), keepdim=True, p=1),
            # Composed  transforms
            tt.Compose(
                [
                    ka.RandomPlasmaBrightness(keepdim=True, p=1),
                    ka.RandomPlasmaContrast(keepdim=True, p=1),
                    ka.RandomPlasmaShadow(shade_intensity=(-0.5, 0.0), keepdim=True, p=1),
                ]
            ),
        ]

        if self.rrc:
            augmentations.append(tt.RandomResizedCrop(224, scale=(0.3, 0.9)))

        if self.jitter:
            augmentations.append(tt.ColorJitter(brightness=0.7, contrast=0.7))

        # Randomly select n augmentations to apply
        chosen_ops = torch.randint(0, len(augmentations), (self.n,))

        # Apply the selected augmentations
        for op_idx in chosen_ops:
            img = augmentations[op_idx](img)

        return img
