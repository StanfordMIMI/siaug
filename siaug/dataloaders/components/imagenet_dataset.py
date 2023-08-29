import os
import re
from glob import glob
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torchvision.datasets import ImageNet


class ImageNetDataset(ImageNet):
    """The (unlabeled) ImageNet dataset with support for resizing, and sampling."""

    def __init__(
        self,
        # return_targets: bool = False,
        img_transform: Optional[Callable] = None,
        lbl_transform: Optional[Callable] = None,
        com_transform: Optional[Callable] = None,
        size: Optional[Tuple[int, int]] = None,
        sample_frac: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.size = size
        self.img_transform = img_transform
        self.lbl_transform = lbl_transform
        self.com_transform = com_transform

        # sanitize arguments
        assert sample_frac is None or (sample_frac > 0.0 and sample_frac <= 1.0)

        self.samples = np.array(
            [(fp, lb) for (fp, lb) in self.samples if not re.search(r"-[0-9]+x[0-9]+.JPEG$", fp)],
            dtype="object",
        )

        if sample_frac is not None:
            # sample %n of the dataset
            idxs = np.arange(len(self.samples))
            idxs = np.random.choice(idxs, int(sample_frac * len(self.samples)), replace=False)
            self.samples = self.samples[idxs]

    def __getitem__(self, idx: int) -> Any:
        path, lbl = self.samples[idx]

        if self.size is not None:
            path = f"-{self.size[0]}x{self.size[1]}".join(os.path.splitext(path))

        # open path as file to avoid ResourceWarning
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")

        if callable(self.img_transform):
            img = self.img_transform(img)

        if callable(self.lbl_transform):
            lbl = self.lbl_transform(lbl)

        sample = {"img": img, "lbl": lbl}
        if callable(self.com_transform):
            return self.com_transform(sample)

        return sample

    @staticmethod
    def save_resized_images(root: os.PathLike, size: Tuple[int, int], splits=["train", "val"]):
        """Save a resized version of the ImageNet dataset."""

        from tqdm.auto import tqdm

        assert size[0] > 0 and size[1] > 0

        for split in splits:
            # find the paths to the original images
            fps = glob(os.path.join(root, f"./{split}/**/*.JPEG"))
            org = [fp for fp in fps if not re.search(r"-[0-9]+x[0-9]+.JPEG$", fp)]

            print(f"Resizing images in the {split} split...")
            for fp in tqdm(org):
                # open the image
                with open(fp, "rb") as f:
                    out_path = f"-{size[0]}x{size[1]}".join(os.path.splitext(fp))
                    if os.path.exists(out_path):
                        continue

                    img = Image.open(f).convert("RGB")
                    img = img.resize(size, resample=Image.Resampling.BICUBIC)
                    img.save(out_path)

        print("Successfully resized images!")
