from neptune.utils import stringify_unsupported
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler

from dataset import KIDataset, train_test_split_stratified
from processor.processor import Leif
from utils.data import binarize


class KIDataModule(LightningDataModule):
    def __init__(self, train_ds=None, val_ds=None, processor_config=None, bundle_as_experiments=False,
                 use_triplets=False, exclude=None, val_size=0.2, binary_classification=False, batch_size=256,
                 num_workers=0, sources=('HC', 'PD_OFF', 'PD_ON'), ips=False):
        """
        :param train_ds: optional prepared train dataset
        :param val_ds: optional prepared validation dataset
        :param processor_config: config for the data processor. If train_ds is not passed, this is a mandatory argument.
        :param bundle_as_experiments: whether to bundle datapoints by experiment
        :param use_triplets: whether to return triplets
        :param exclude: a list of data categories to exclude
        :param val_size: the relative size of the validation split
        :param binary_classification: whether to binarize the labels
        :param batch_size: the size of mini-batches, if -1 then equal to the length of the dataset
        :param num_workers: the number of workers for the dataloaders
        :param sources: the data sources to use
        :param ips: whether to use interpersonal sampling
        """
        super().__init__()
        self.train_ds, self.val_ds, self.test_ds = None, None, None
        if train_ds is not None:
            self.train_ds = train_ds
            self.train_ds.use_triplets = use_triplets
        if val_ds is not None:
            self.val_ds = val_ds
            self.val_ds.use_triplets = use_triplets

        self.bundle_as_experiments = bundle_as_experiments
        self.use_triplets = use_triplets
        self.exclude = exclude
        self.val_size = val_size
        self.binary_classification = binary_classification
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sources = sources
        self.ips = ips

        if processor_config is not None:
            self.processor = Leif(processor_config)
            stringified_processor_config = stringify_unsupported(processor_config)
            self.save_hyperparameters(stringified_processor_config)

        # If prepared datasets were passed to init, don't log them as hyperparameters.
        # Likewise, don't log the processor config, since it holds unsupported parameter types. (handled above)
        self.save_hyperparameters(ignore=['train_ds', 'val_ds', 'test_ds', 'processor_config'])

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" and self.train_ds is None:
            train_val_ds = KIDataset(data_processor=self.processor,
                                     train=True,
                                     bundle_as_sessions=self.bundle_as_experiments,
                                     use_triplets=self.use_triplets,
                                     exclude=self.exclude,
                                     sources=self.sources,
                                     ips=self.ips)
            if self.val_size != 0:
                self.train_ds, self.val_ds = train_test_split_stratified(train_val_ds, test_size=self.val_size)
            else:
                self.train_ds = train_val_ds

        # Assign test dataset for use in dataloader
        if stage == "test":
            self.test_ds = KIDataset(data_processor=self.processor,
                                     train=False,
                                     bundle_as_sessions=self.bundle_as_experiments,
                                     use_triplets=False,
                                     exclude=self.exclude,
                                     sources=self.sources)

        # Binarize dataset after split to make sure split is stratified w.r.t all three classes
        if self.binary_classification:
            for ds in [self.train_ds, self.val_ds, self.test_ds]:
                if ds is not None:
                    binarize(ds)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=len(self.train_ds) if self.batch_size == -1 else self.batch_size,
                          sampler=ImbalancedDatasetSampler(self.train_ds, callback_get_label=lambda item: item.y),
                          num_workers=self.num_workers)

    def val_dataloader(self):
        if self.val_ds is None:
            return None
        return DataLoader(self.val_ds, batch_size=len(self.val_ds) if self.batch_size == -1 else self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=len(self.test_ds) if self.batch_size == -1 else self.batch_size,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return self.test_dataloader()

    def set_use_triplets(self, value: bool):
        self.train_ds.use_triplets = value
        if self.val_ds is not None:
            self.val_ds.use_triplets = value

    # Flattens the datasets, i.e. removes the session dimension
    def flatten(self):
        self.train_ds.flatten()
        self.val_ds.flatten()
        if self.test_ds is not None:
            self.test_ds.flatten()

    # Batches the datasets, i.e. adds the session dimension
    def batch(self):
        self.train_ds.batch()
        self.val_ds.batch()
        if self.test_ds is not None:
            self.test_ds.batch()

    def class_names(self):
        return ['HC', 'PD OFF', 'PD ON'] if not self.binary_classification else ['HC', 'PD']

    @staticmethod
    def group_names():
        return ['HC', 'PD OFF', 'PD ON']
