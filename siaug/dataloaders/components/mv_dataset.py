import json
import multiprocessing as mp
import os
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import dosma as dm
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map


def _extract_idxs(
    fname: str,
    data_dir: os.PathLike,
    load: Callable[[os.PathLike], dm.MedicalVolume],
    axis: int,
) -> List[Tuple[str, int]]:
    """Pickle-able/parallelizable utility function that creates a list of indices given an axis."""

    mv = load(os.path.join(data_dir, fname))
    return [(fname, idx) for idx in range(mv.shape[axis])]


class MVDataset(Dataset):
    """PyTorch dataset that operates on MedicalVolumes and supports multiple iteration modes.

    TODO: replace data_dir with fn that returns an iterable with paths
    TODO: allow for multiple return types from the __getitem__ classmethod or multiple from the loader
    TODO: provide support for caching
    """

    def __init__(
        self,
        data_dir: os.PathLike,
        loader: Callable[[str], dm.MedicalVolume],
        split: Optional[Union[os.PathLike, Callable]] = None,
        plane: Optional[str] = None,
        transform: Optional[Callable] = None,
        sample_n: Optional[int] = None,
        sample_frac: Optional[float] = None,
        num_workers: int = 1,
        verbose: bool = True,
    ):
        # setup
        self.data_dir = data_dir
        self.loader = loader
        self.split = split
        self.plane = plane
        self.transform = transform
        self.num_workers = num_workers
        self.verbose = verbose
        self.axis = self.to_axis(plane)

        # generate the data split
        self.samples = self._get_split()

        # subsample
        if sample_n is not None:
            self.samples = self.sample_n(self.samples, sample_n)

        if sample_frac is not None:
            self.samples = self.sample_frac(self.samples, sample_frac)

    def _make_dataset(self, axis: int = None, num_workers: int = 1) -> List:
        """Create a dataset from a directory and a given axis."""

        fnames = os.listdir(self.data_dir)
        if axis is None:
            return [(fname, None) for fname in fnames]

        extract = partial(_extract_idxs, data_dir=self.data_dir, load=self.loader, axis=axis)

        if self.verbose:
            samples = process_map(extract, fnames, max_workers=num_workers, chunksize=1)
        else:
            with mp.Pool(self.num_workers) as p:
                samples = p.map(extract, fnames)

        # flatten the samples
        return [sample for sublist in samples for sample in sublist]

    def _get_split(self):
        """Construct a split with a list of samples."""

        if isinstance(self.split, str):
            with open(self.split) as f:
                return json.load(f)

        if callable(self.split):
            return self.split(self.data_dir)

        return self._make_dataset(self.axis, self.num_workers)

    def _get_instance(self, idx: int) -> os.PathLike:
        """Retrieve a MedicalVolume with an index and an axis."""

        fname, slice_idx = self.samples[idx]
        path = os.path.join(self.data_dir, fname)
        mv = self.loader(path)

        # store pointers on the medical volume
        # TODO: change offset to indices along each axis
        # TODO: implement these properties on MV itself
        mv.index = slice_idx
        mv.axis = self.axis

        return mv

    def __len__(self) -> int:
        """Get the length of the dataset."""

        return len(self.samples)

    def __getitem__(self, idx: int):
        """Get a MedicalVolume and corresponding metadata."""

        if self.transform:
            return self.transform(self._get_instance(idx))

        return self._get_instance(idx)

    def save_split(self, path: os.PathLike):
        """Save the current split to disk."""

        with open(path) as f:
            json.dump(self.samples, f)

    def compute_normalization_constants(self, limit: int = None) -> Tuple[float, float]:
        """Compute the normalization constants for a dataset."""

        assert (
            self.transform is not None
        ), "The transformation pipeline should output an array-like."

        arr = []
        length = len(self) if not limit else limit
        for idx, item in enumerate(tqdm(self, total=length, disable=not self.verbose)):
            if idx == limit:
                break

            arr.append(np.ravel(item))

        arr = np.concatenate(arr)
        return (np.mean(arr), np.std(arr))

    @staticmethod
    def to_axis(plane: str) -> Optional[str]:
        """Convert plane to axis name."""

        assert plane is None or plane in ["axial", "z", "coronal", "y", "saggital", "x"]
        return (
            {
                "axial": -1,
                "z": -1,
                "coronal": -3,
                "y": -3,
                "saggital": -2,
                "x": -2,
            }
        ).get(plane, None)

    @staticmethod
    def sample_n(samples: List, n: int) -> List:
        """Sample n samples."""

        assert isinstance(n, int) and n > 0

        idxs = np.random.choice(np.arange(len(samples)), size=n, replace=False)
        return [samples[idx] for idx in idxs]

    @classmethod
    def sample_frac(cls, samples: List, frac: float) -> List:
        """Sample a fraction of the samples."""

        assert frac > 0 and frac <= 1

        num_samples = int(frac * len(samples))
        return cls.sample_n(samples=samples, n=num_samples)
