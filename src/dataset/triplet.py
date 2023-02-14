from typing import Tuple

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from dataset.ki import KIDataset


class TripletSegmentDataset(Dataset):
    def __init__(self, ds: KIDataset, binary_clf: bool):
        self.ds = ds
        self.binary_clf = binary_clf

    def __process_label(self, lab: torch.Tensor) -> torch.Tensor:
        if self.binary_clf:
            if lab == 2:
                lab = torch.tensor(1, dtype=int)
            return F.one_hot(lab.type(torch.int64), num_classes=2 if self.binary_clf else 3)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor] or Tuple[torch.Tensor, ...]:
        anchor = self.ds[idx]
        if not self.ds.train:
            return anchor.x, self.__process_label(anchor.y)

        # Augmentation: add light random noise
        anchor.x += 1e-2 * torch.randn_like(anchor.x)

        positive_indices = (self.ds.y == anchor.y).nonzero()
        positive_indices = positive_indices[positive_indices != idx][:, None]
        negative_indices = (self.ds.y != anchor.y).nonzero()

        positive_idx = torch.randperm(positive_indices.numel())[0]
        negative_idx = torch.randperm(negative_indices.numel())[0]

        positive = self.ds[positive_idx]
        negative = self.ds[negative_idx]

        return anchor.x, self.__process_label(anchor.y), positive.x, self.__process_label(
            positive.y), negative.x, self.__process_label(negative.y)

    def __len__(self) -> int:
        return self.ds.__len__()
