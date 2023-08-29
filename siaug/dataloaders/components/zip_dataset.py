from typing import Callable, List, Optional

from torch.utils.data import Dataset


class ZipDataset(Dataset):
    """Dataset wrapping datasets.

    Each sample will be retrieved as a tuple of values indexed from each dataset individually.

    Args:
        datasets (sequence): Lists of datasets to be concatenated.
        transform (callable, optional): Transforms to apply on the tuple of outputs.
    """

    def __init__(
        self,
        datasets: List[Dataset],
        transform: Optional[Callable] = None,
        truncate_datasets: bool = False,
    ):
        if not truncate_datasets:
            assert all(
                [len(ds) == datasets[0] for ds in datasets]
            ), "Datasets must be of equal length."

        self.datasets = datasets
        self.transform = transform

    def __getitem__(self, i):
        if callable(self.transform):
            self.transform(tuple(ds[i] for ds in self.datasets))

        return tuple(ds[i] for ds in self.datasets)

    def __len__(self):
        return min(len(ds) for ds in self.datasets)
