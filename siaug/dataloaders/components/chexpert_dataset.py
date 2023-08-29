import os
from typing import Callable, List, Tuple
import re

import numpy as np
import polars as pl
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from siaug.utils.dataset import CHEXPERT_PATHOLOGIES, preprocess_chexpert_labels


class CheXpertDataset(Dataset):
    """The CheXpert-small (JPG) dataset.

    This dataset supports splits with text reports, if formatted in the same way as the
    original CheXpert dataset (columns need to be appended to the end of the CSV file).

    Args:
        path (os.PathLike): Path to the CSV file, typically a split.
        data_dir (os.PathLike): Path to the data directory.
        img_transform (Callable, optional): Image transform. Defaults to None.
        txt_transform (Callable, optional): Text transform. Defaults to None.
        lbl_transform (Callable, optional): Label transform. Defaults to None.
        com_transform (Callable, optional): Composite transform. Defaults to None.
        columns (List[str], optional): List of columns to use as labels.
        txt_key (str, optional): Key to use for the text report. Defaults to "report_impression".
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
        txt_key: str = None,
        na_mode: str = None,
        verbose: bool = True,
    ):
        self.img_transform = img_transform
        self.txt_transform = txt_transform
        self.lbl_transform = lbl_transform
        self.com_transform = com_transform

        self.make_dataset(path, data_dir, columns, na_mode, txt_key)

        self.verbose = verbose

        

    def make_dataset(
        self,
        path: os.PathLike,
        data_dir: os.PathLike,
        columns: List[str],
        na_mode: str,
        txt_key: str,
    ):
        self.df = pl.read_csv(
            path, dtypes=[pl.Utf8] * 2 + [pl.Int32] + [pl.Utf8] * 2 + [pl.Float32] * 14
        )
        
        # rename columns
        col_mapping = dict(
            zip(self.df.columns, [re.sub(r"\s+|\/", "_", col.lower()) for col in self.df.columns])
        )

        self.df = self.df.rename(col_mapping)

        imgs = self.df.get_column("path").apply(lambda x: os.path.join(data_dir, x[20:])).to_list()
        txts = self.df.get_column(txt_key).to_list() if txt_key else [None] * len(imgs)
        lbls = preprocess_chexpert_labels(self.df, columns, na_mode)
        self.samples = list(zip(imgs, txts, lbls))

    def __getitem__(self, idx: int):
        img, txt, lbl = self.samples[idx]
        img = Image.open(img)
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
        """Compute the normalization constants for a dataset."""

        arr = []
        length = len(self) if not limit else limit
        for idx in tqdm(range(length), disable=not self.verbose):
            arr.append(np.ravel(self[idx][0]))

        arr = np.concatenate(arr)
        return (np.mean(arr), np.std(arr))
