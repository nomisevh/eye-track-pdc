import sys
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Any

import numpy as np
from numpy import array
from sklearn.model_selection import train_test_split
from torch import cat, Tensor
from torch.utils.data import Dataset
from torchmetrics.functional.classification import binary_accuracy

from src.utils.misc import set_random_state
from utils.path import log_path


def concat_datasets(datasets: Sequence[Dataset]):
    cls = datasets[0].__class__
    joint_dataset = cls.__new__(cls)
    for attr in datasets[0].__dict__.keys():
        if isinstance(getattr(datasets[0], attr), Tensor):
            setattr(joint_dataset, attr, cat([getattr(dataset, attr) for dataset in datasets], dim=0))
        else:
            # For attributes that are not tensors, just use the value from the first dataset
            setattr(joint_dataset, attr, getattr(datasets[0], attr))
    return joint_dataset


def k_fold_cross_validator(dataset: Dataset, k: int, splitter=None) -> Tuple[Dataset, Dataset]:
    if splitter is None:
        splitter = train_test_split

    # Generate folds first, to ensure they are non-overlapping
    folds = []
    remainder = dataset
    for i in range(k):
        test_fraction = (1 / (k - i))
        # Last fold
        if test_fraction == 1:
            folds.append(remainder)
            break

        # Split into fold and remainder
        remainder, fold = splitter(remainder, test_size=test_fraction)
        folds.append(fold)

    for i in range(len(folds)):
        # Let fold be the test split
        test_ds = folds[i]
        # Let all other folds be the train split
        train_ds = concat_datasets([fold for j, fold in enumerate(folds) if i != j])

        yield train_ds, test_ds


class Validator:
    """
    Validates a result by running a callback function with stratified n-fold cross validation for various random seeds.
    """

    SEEDS = np.arange(100)

    class Callback(ABC):
        @abstractmethod
        def __call__(self, train_ds: Dataset, val_ds: Dataset, iteration: int, **kwargs) -> Any:
            raise NotImplementedError

    def __init__(self, num_random_inits=3, num_folds=5, splitter=None):
        """
        :param num_random_inits: The number of unique random seeds to run
        :param num_folds: The number of folds to use in cross fold validation
        :param splitter: The function to use for splitting the data during cross validation. Should satisfy the same
        interface as sklearn.model_selection.train_test_split.
        """
        self.num_random_inits = num_random_inits
        self.num_folds = num_folds
        self.splitter = splitter

    def __call__(self, f: Callback, dataset: Dataset, **bits):
        """
        Will validate the function f for various random initializations and train/val splits. The number of runs is
        num_random_inits * num_folds.

        :param f: The function to be run.
        :param dataset: The dataset to be used in cross validation. This is usually the joint train/val set.
        :param bits: Keyword arguments to be drilled to f.
        :return: The average of the metric returned by f, over all random initializations and all folds.
        """
        metrics = []
        iteration = 0
        for seed in self.SEEDS[:self.num_random_inits]:
            set_random_state(seed)
            for train_ds, val_ds in k_fold_cross_validator(dataset, k=self.num_folds, splitter=self.splitter):
                out = f(train_ds, val_ds, iteration, **bits)
                metrics.append(out)
                iteration += 1

        return np.average(metrics), np.std(metrics)


def grid_search_2d(validator: Validator, callback: Validator.Callback, dataset: Dataset, **kwargs):
    (p1, p1_candidates), (p2, p2_candidates) = kwargs.items()

    scores = np.zeros((len(p1_candidates), len(p2_candidates)))
    deviations = np.zeros((len(p1_candidates), len(p2_candidates)))
    tot_configs = scores.size

    for i, p1_val in enumerate(p1_candidates):
        for j, p2_val in enumerate(p2_candidates):
            print(f'searching config {j + i * len(p2_candidates) + 1} / {tot_configs} | {p1}={p1_val}, {p2}={p2_val}',
                  file=sys.stderr)

            scores[i, j], deviations[i, j] = validator(callback, dataset, **{p1: p1_val, p2: p2_val})

    np.savetxt(log_path.joinpath('grid_search_result_mean.csv'), scores, delimiter=',')  # noqa
    np.savetxt(log_path.joinpath('grid_search_result_std.csv'), deviations, delimiter=',')  # noqa

    return scores


def get_attribute_power(batch, pred, threshold=0.5):
    return {
        'prosaccade': compute_attribute_power(1 - batch.s, pred, batch.y, threshold=threshold),
        'antisaccade': compute_attribute_power(batch.s, pred, batch.y, threshold=threshold),
        'dopamine': compute_attribute_power(array(batch.g != 1, dtype=int), pred, batch.y, threshold=threshold),
        'non-dopamine': compute_attribute_power(array(batch.g != 2, dtype=int), pred, batch.y, threshold=threshold),
    }


def compute_attribute_power(attribute, preds, targets, threshold):
    subgroup_acc = binary_accuracy(preds[attribute == 1], targets[attribute == 1].long(), threshold=threshold).round(
        decimals=4).item()
    all_data_acc = binary_accuracy(preds, targets.long(), threshold=threshold).round(decimals=4).item()
    return subgroup_acc - all_data_acc
