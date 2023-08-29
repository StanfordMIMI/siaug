import os
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from siaug.utils.dataset import CHEXPERT_PATHOLOGIES, VINDR_PATHOLOGIES, preprocess_chexpert_labels, preprocess_vindr_filepaths

__all__ = ["VinDRDataset"]


class VinDRDataset(Dataset):
    """The VinDR (DICOM) dataset.

    This dataset contains 18,000 chest X-ray images of patients (15,000 in train and 3000 in test). Patient metadata is not available. The images in this dataset are expected to be saved as Zarr files in ZipStores. Metadata is available in a CSV file.

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
        columns: List[str] = VINDR_PATHOLOGIES,
        na_mode: str = None,
        verbose: bool = True,
    ):
        self.img_transform = img_transform
        self.txt_transform = txt_transform
        self.lbl_transform = lbl_transform
        self.com_transform = com_transform
        self.verbose = verbose

        if self.txt_transform:
            ValueError("Text transform is not supported for this dataset.")

        self.make_dataset(path, data_dir, columns, na_mode)

    def make_dataset(
        self,
        path: os.PathLike,
        data_dir: os.PathLike,
        columns: List[str],
        na_mode: str,
    ):

        assert "ground_truth" in str(path), "Path should contain 'ground_truth'. Please preprocess the data first using the preprocessing script `preprocess_vindr_csvs` in `siaug.utils.dataset`"

        self.df = pl.read_csv(path, infer_schema_length=1000)
        imgs = self.df['image_path'].to_list()

        lbls = preprocess_chexpert_labels(self.df, columns, na_mode)
        self.samples = list(zip(imgs, lbls))

    def __getitem__(self, idx: int):
        img, lbl = self.samples[idx]
        sample = {"img": img, "lbl": lbl}

        if callable(self.img_transform):
            sample["img"] = self.img_transform(sample["img"])

        if callable(self.lbl_transform):
            sample["lbl"] = self.lbl_transform(sample["lbl"])

        if callable(self.com_transform):
            return self.com_transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)

    def compute_normalization_constants(self, limit: int = None) -> Tuple[float, float]:
        """Compute the normalization constants for VinDR."""
        arr = []
        length = len(self) if not limit else limit
        for idx in tqdm(range(length), disable=not self.verbose):
            arr.append(np.ravel(self[idx][0]))

        arr = np.concatenate(arr)
        return (np.mean(arr), np.std(arr))
