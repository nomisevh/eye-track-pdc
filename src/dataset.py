from collections import namedtuple
from os import makedirs
from os.path import exists, isfile
from typing import Tuple, Sequence

from numpy import array
from sklearn.model_selection import train_test_split
from torch import tensor, int as torch_int, save, load, isin, logical_not, cat
from torch.utils.data import Dataset
from yaml import load as load_yaml, FullLoader

from processor.interface import MainProcessor
from processor.processor import Leif
from utils.ki import LABELS as KI_LABELS, FILENAME_REGEX as KI_FILENAME_REGEX, AXIS as KI_AXIS, SACCADE as KI_SACCADE, \
    load_data
from utils.misc import torch_unique_index
from utils.path import ki_data_tmp_path, config_path


class KIDataset(Dataset):
    Signature = namedtuple('Signature', ['x', 'y', 'z', 'r', 'a', 's'])

    def __init__(self, *, data_processor: MainProcessor, train: bool):
        file_path = ki_data_tmp_path.joinpath(f"ki-dataset-{'train' if train else 'test'}")

        # Use cached version of dataset if available
        if isfile(file_path):
            self.__dict__.update(load(file_path))
            print(f'loaded dataset from {file_path}')
            return

        dataframes, filenames = load_data(train)

        segmented_files = data_processor(dataframes, train=train)

        x, y, z, r, a, s = populate_ki(segmented_files, filenames)

        # Tensor with shape (N, M, T) holding the multivariate time series.
        # N is number of data points, M is the dimensionality and T is the length of the series.
        self.x = tensor(array(x)).float().permute(0, 2, 1)
        # Tensor with shape (N) holding the labels
        self.y = tensor(y).float()
        # Tensor with shape (N) holding which patient the data points belong to
        self.z = tensor(z, dtype=torch_int)
        # Tensor with shape (N) holding which trial each segment belongs to
        self.r = tensor(r, dtype=torch_int)
        # Tensor with shape (N) holding the axis each segment is aligned with (0:'horiz', 1:'vert')
        self.a = tensor(a, dtype=torch_int)
        # Tensor with shape (N) holding the saccade type of each segment (0:'pro', 1:'anti')
        self.s = tensor(s, dtype=torch_int)

        # Cache data for faster future loading
        self.save(file_path)

    def __getitem__(self, item):
        return self.Signature(self.x[item], self.y[item], self.z[item], self.r[item], self.a[item], self.s[item])

    def __len__(self):
        return len(self.y)

    def save(self, save_path):
        if not exists(save_path.parent):
            makedirs(save_path.parent)
        save({'x': self.x, 'y': self.y, 'z': self.z, 'r': self.r, 'a': self.a, 's': self.s}, save_path)

    def clone(self, index):
        clone = KIDataset.__new__(KIDataset)
        clone.__dict__.update({attr: value.clone()[index] for attr, value in self.__dict__.items()})
        return clone

    @classmethod
    def concat(cls, datasets: Sequence['KIDataset']):
        joint_dataset = cls.__new__(cls)
        for attr in datasets[0].__dict__.keys():
            setattr(joint_dataset, attr, cat([getattr(dataset, attr) for dataset in datasets], dim=0))
        return joint_dataset


def populate_ki(segmented_files, filenames):
    datapoints = []
    for trial, (segments, filename) in enumerate(zip(segmented_files, filenames)):
        individual, group, axis, saccade = KI_FILENAME_REGEX.findall(filename)[0]
        for seg in segments:
            datapoints.append(KIDataset.Signature(
                x=seg.values,
                y=KI_LABELS[group],
                z=int(individual),
                r=trial,
                a=KI_AXIS[axis],
                s=KI_SACCADE[saccade],
            ))
    return zip(*datapoints)


def train_test_split_stratified(dataset: KIDataset, test_size: float = 0.1, seed=42):
    """
    Split KIDataset instance's data into training and test set satisfying two conditions:
        - mutually exclusive sets of patients
        - (approximately) equal class distributions in each subset
    :param KIDataset dataset: dataset instance
    :param float test_size: test set size percentage (defaults to 0.1 or 10%)
    :param int seed: SRNG seed (defaults to 42)
    :return: a tuple object with two new KIDataset instances
    """
    # Sample Patient Numbers
    zs, zsi = torch_unique_index(dataset.z)
    ys = dataset.y[zsi]
    z_train, z_test, y_train, y_test = train_test_split(zs, ys, test_size=test_size, stratify=ys, random_state=seed)
    # Create Subsets
    train_indices = isin(dataset.z, z_train)
    return dataset.clone(train_indices), dataset.clone(logical_not(train_indices))


def k_fold_cross_validator(dataset: KIDataset, k: int) -> Tuple[KIDataset, KIDataset]:
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
        remainder, fold = train_test_split_stratified(remainder, test_size=test_fraction)
        folds.append(fold)

    for i in range(len(folds)):
        # Let fold be the train split
        train_ds = folds[i]
        # Let all other folds be the test split
        test_ds = KIDataset.concat([fold for j, fold in enumerate(folds) if i != j])

        yield train_ds, test_ds


def test():
    with open(f'{config_path}/leif.yaml', 'r') as reader:
        config = load_yaml(reader, Loader=FullLoader)

    processor = Leif(config)

    ds = KIDataset(data_processor=processor, train=True)

    for train_ds, test_ds in k_fold_cross_validator(ds, k=4):
        ...


if __name__ == '__main__':
    test()
