from typing import Tuple

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from dataset.ki import KIDataset


class IndividualTrialDataset(Dataset):
    def __init__(self, ds: KIDataset, binary_clf: bool = True):
        self.ds = ds
        self.binary_clf = binary_clf

    def __process_label(self, lab: torch.Tensor) -> torch.Tensor:
        if self.binary_clf:
            if lab == 2:
                lab = torch.tensor(1, dtype=int)
            return F.one_hot(lab.type(torch.int64), num_classes=2 if self.binary_clf else 3)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            trial_data = self.ds[idx]
            return torch.stack([s.x for s in trial_data.x.usable_segments]), self.__process_label(trial_data.y)
        except RuntimeError:
            return self.__getitem__((idx + 1) % len(self))

    def __len__(self) -> int:
        return self.ds.__len__()
