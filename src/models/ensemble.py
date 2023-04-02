from typing import Sequence

from pytorch_lightning import LightningModule
from torch import nn


class Ensemble(LightningModule):
    def __init__(self, models: Sequence[nn.Module], lr: float, wd: float):
        super().__init__()
