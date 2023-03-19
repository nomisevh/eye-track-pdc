from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler

from dataset import KIDataset, train_test_split_stratified
from processor.processor import Leif
from utils.data import binarize


class KIDataModule(LightningDataModule):
    def __init__(self, train_ds=None, val_ds=None, processor_config=None, bundle_as_trials=False, use_triplets=False,
                 val_size=0.2, binary_classification=False, batch_size=256):
        """
        :param train_ds: optional prepared train dataset
        :param val_ds: optional prepared validation dataset
        :param processor_config: config for the data processor
        :param bundle_as_trials: whether to bundle datapoints by trials
        :param use_triplets: whether to return triplets
        :param val_size: the relative size of the validation split
        :param binary_classification: whether to binarize the labels
        :param batch_size: the size of mini-batches, if -1 then equal to the length of the dataset
        """
        super().__init__()
        self.train_ds = train_ds if train_ds is not None else None
        self.val_ds = val_ds if val_ds is not None else None
        self.test_ds = None
        self.bundle_as_trials = bundle_as_trials
        self.use_triplets = use_triplets
        self.val_size = val_size
        self.binary_classification = binary_classification
        self.batch_size = batch_size

        self.processor = Leif(processor_config)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            train_val_ds = KIDataset(data_processor=self.processor,
                                     train=True,
                                     bundle_as_trials=self.bundle_as_trials,
                                     use_triplets=self.use_triplets)
            self.train_ds, self.val_ds = train_test_split_stratified(train_val_ds, test_size=self.val_size)

        # Assign test dataset for use in dataloader
        if stage == "test":
            self.test_ds = KIDataset(data_processor=self.processor,
                                     train=False,
                                     bundle_as_trials=self.bundle_as_trials,
                                     use_triplets=False)

        # Binarize dataset after split to make sure split is stratified w.r.t all three classes
        if self.binary_classification:
            for ds in [self.train_ds, self.val_ds, self.test_ds]:
                if ds is not None:
                    binarize(ds)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=len(self.train_ds) if self.batch_size == -1 else self.batch_size,
                          sampler=ImbalancedDatasetSampler(self.train_ds, callback_get_label=lambda item: item.y))

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=len(self.val_ds) if self.batch_size == -1 else self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=len(self.test_ds) if self.batch_size == -1 else self.batch_size)

    def set_use_triplets(self, val: bool):
        self.train_ds.use_triplets = val
        self.val_ds.use_triplets = val
