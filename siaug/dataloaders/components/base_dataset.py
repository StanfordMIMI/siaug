from typing import Any, Callable

from torch.utils.data import Dataset

__all__ = ["BaseDataset"]


class BaseDataset(Dataset):
    def __init__(self, data: Any, key: str, transform: Callable = None):
        self.data, self.key, self.transform = data, key, transform

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        if callable(self.transform):
            return self.transform(sample)

        return {self.key: sample}

    def __len__(self):
        return len(self.data)
