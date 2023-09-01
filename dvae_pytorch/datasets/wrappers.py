from collections.abc import Sized
from typing import cast

import torch
from torch.utils.data import Dataset


class ClassificationDatasetWrapper(Dataset):
    """A wrapper for classification datasets that makes them usable with Auto Encoders.

    The class acts as a proxy to the passed dataset, but does not return
    labels from the original dataset, since Auto Encoders do not need them.
    """

    def __init__(self, dataset: Dataset) -> None:  # noqa: D107
        self.dataset = dataset

    def __getitem__(self, item: int) -> torch.Tensor:  # noqa: D105
        return self.dataset[item][0]

    def __len__(self) -> int:  # noqa: D105
        return len(cast(Sized, self.dataset))
