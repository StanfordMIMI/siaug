import os
from typing import Callable, List, Tuple

import numpy as np
import polars as pl
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from siaug.utils.dataset import CHEXPERT_PATHOLOGIES, preprocess_chexpert_labels

__all__ = ["MimicDataset"]


class MimicDataset(Dataset):
    """The MIMIC-CXR (DICOM) dataset.

    This dataset contains 377,110 chest X-ray images of 227,835 unique patients.
    The images in this dataset are expected to be saved as Zarr files in
    ZipStores.

    Args:
        path (os.PathLike): Path to the CSV file, typically a split.
        data_dir (os.PathLike): Path to the directory containing the Zarr files.
        img_transform (Callable, optional): Image transform. Defaults to None.
        txt_transform (Callable, optional): Text transform. Defaults to None.
        lbl_transform (Callable, optional): Label transform. Defaults to None.
        com_transform (Callable, optional): Composite transform. Defaults to None.
        columns (List[str], optional): List of columns to use as labels.
        na_mode (NaMode, optional): NA mode. Defaults to NaMode.NONE.
        verbose (bool, optional): Whether to show a progress bar. Defaults to True.
    """

    def __init__(
        self,
        path: os.PathLike,
        data_dir: os.PathLike,
        img_transform: Callable = None,
        txt_transform: Callable = None,
        lbl_transform: Callable = None,
        com_transform: Callable = None,
        columns: List[str] = CHEXPERT_PATHOLOGIES,
        na_mode: str = None,
        verbose: bool = True,
    ):
        self.img_transform = img_transform
        self.txt_transform = txt_transform
        self.lbl_transform = lbl_transform
        self.com_transform = com_transform
        self.verbose = verbose
        self.make_dataset(path, data_dir, columns, na_mode)

    def make_dataset(
        self,
        path: os.PathLike,
        data_dir: os.PathLike,
        columns: List[str],
        na_mode: str,
    ):
        self.df = pl.read_csv(path, infer_schema_length=1000).rename(
            {"Pleural Effusion": "Effusion"}
        )
        imgs = (
            self.df.select(["group", "dicom_id"])
            .apply(lambda x: os.path.join(data_dir, "volumes", str(x[0]), x[1] + ".pt"))
            .to_series()
            .to_list()
        )

        txts = (
            self.df.get_column("study_id")
            .apply(lambda x: os.path.join(data_dir, "reports", f"{x}.txt"))
            .to_list()
        )

        lbls = preprocess_chexpert_labels(self.df, columns, na_mode)
        self.samples = list(zip(imgs, txts, lbls))

    def __getitem__(self, idx: int):
        img, txt, lbl = self.samples[idx]
        sample = {"img": img, "txt": txt, "lbl": lbl}

        if callable(self.img_transform):
            sample["img"] = self.img_transform(sample["img"])

        if callable(self.txt_transform):
            sample["txt"] = self.txt_transform(sample["txt"])

        if callable(self.lbl_transform):
            sample["lbl"] = self.lbl_transform(sample["lbl"])

        if callable(self.com_transform):
            return self.com_transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)

    def compute_normalization_constants(self, limit: int = None) -> Tuple[float, float]:
        """Compute the normalization constants for MIMIC."""
        arr = []
        length = len(self) if not limit else limit
        for idx in tqdm(range(length), disable=not self.verbose):
            arr.append(np.ravel(self[idx][0]))

        arr = np.concatenate(arr)
        return (np.mean(arr), np.std(arr))
