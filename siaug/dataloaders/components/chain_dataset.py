from collections import ChainMap
from typing import Callable, List

from torch.utils.data import Dataset

from siaug.dataloaders.components.base_dataset import BaseDataset

__all__ = ["ChainDataset"]


class ChainDataset(BaseDataset):
    def __init__(self, datasets: List[Dataset], transform: Callable = None):
        self.datasets, self.transform = datasets, transform

    def __getitem__(self, idx) -> ChainMap:
        sample = ChainMap(*[ds[idx] for ds in self.datasets])
        if callable(self.transform):
            return self.transform(sample)

        return sample

    def __len__(self):
        return min(len(ds) for ds in self.datasets)
