import random
from collections import namedtuple
from os import makedirs
from os.path import exists, isfile

from numpy import array
from sklearn.model_selection import train_test_split
from torch import tensor, int as torch_int, bool as torch_bool, save, load, isin, logical_not, argwhere, \
    Tensor, zeros, logical_or, argsort, bincount, stack, logical_and
from torch.utils.data import Dataset
from yaml import load as load_yaml, FullLoader

from processor.interface import MainProcessor
from processor.processor import Leif
from utils.data import binarize
from utils.interpretability import compute_average_power_spectrum, plot_power_spectrum
from utils.ki import LABELS as KI_LABELS, FILENAME_REGEX as KI_FILENAME_REGEX, AXIS as KI_AXIS, SACCADE as KI_SACCADE, \
    load_data, SAMPLE_RATE
from utils.misc import torch_unique_index
from utils.path import ki_data_tmp_path, config_path
from utils.visualize import plot_trial

Signature = namedtuple('Signature', ['x', 'y', 'z', 'r', 'a', 's', 'g'])


class KIDataset(Dataset):
    # All the attributes that have a single value per data point
    SINGULAR_ATTRIBUTES = [attr for attr in Signature._fields if attr != 'x']

    def __init__(self, *, data_processor: MainProcessor, train: bool, bundle_as_sessions=False, use_triplets=False,
                 exclude=None, sources=('HC', 'PD_OFF', 'PD_ON'), ips: bool = False):
        """
        :param data_processor: The data processor to use for processing the data
        :param train: Whether to use the train or test set
        :param bundle_as_sessions: Whether to bundle the data as sessions
        :param use_triplets: Whether to return triplets
        :param exclude: List of data categories to exclude
        :param sources: List of data sources to use
        :param ips: Whether to use interpersonal sampling
        """

        self.use_triplets = use_triplets
        self.ips = ips

        file_path = ki_data_tmp_path.joinpath(self.format_filename(train, bundle_as_sessions, sources))

        # Use cached version of dataset if available
        if isfile(file_path):
            self.__dict__.update(load(file_path))
            print(f'loaded dataset from {file_path}')
        else:
            dataframes, filenames = load_data(train, sources)

            segmented_files, kept = data_processor(dataframes)
            # Only keep the filenames of the files that were not filtered out
            filenames = array(filenames)[kept]

            if bundle_as_sessions:
                x, y, z, r, a, s, g = populate_ki_sessions(segmented_files, filenames)
                # Tensor with shape (N, L, M, T) holding the multivariate time series.
                # N is number of sessions, L is the number of trials in each session, M is the dimensionality and
                # T is the length of the series.
                self.x = tensor(array(x)).float().permute(0, 1, 3, 2)
            else:
                x, y, z, r, a, s, g = populate_ki_trials(segmented_files, filenames)
                # Tensor with shape (N, M, T) holding the multivariate time series.
                # N is number of trials, M is the dimensionality and T is the length of the series.
                self.x = tensor(array(x)).float().permute(0, 2, 1)

            # Tensor with shape (N) holding the labels
            self.y = tensor(y).float()
            # Tensor with shape (N) holding which individual the data points belong to
            self.z = tensor(z, dtype=torch_int)
            # Tensor with shape (N) holding which session each trial belongs to
            self.r = tensor(r, dtype=torch_int)
            # Tensor with shape (N) holding the axis each trial is aligned with (0:'horiz', 1:'vert')
            self.a = tensor(a, dtype=torch_int)
            # Tensor with shape (N) holding the saccade type of each trial (0:'pro', 1:'anti')
            self.s = tensor(s, dtype=torch_int)
            # Tensor with shape (N) holding the group each trial belongs to (0:'HC', 1:'PDOFF', 2:'PDON')
            # Similar to y but is not affected by binarization.
            self.g = tensor(g, dtype=torch_int)

            # Cache data for faster future loading
            self.save(file_path)

        # OBS currently excluding data after sanitization, meaning that sanitization is still affected by excluded data.
        # Exclude data as requested
        if exclude is not None:
            self.exclude_data(exclude)

    def __getitem__(self, item):
        anchor = self._get(item)
        if not self.use_triplets:
            return anchor

        if self.ips:
            # Compute the indices of all items that have the same label but are from a different individual than the
            # anchor
            positive_indices = argwhere(logical_and(self.y == anchor.y, self.z != anchor.z))  # noqa
        else:
            # Compute the indices of all items that have the same label as the anchor
            positive_indices = argwhere(self.y == anchor.y)  # noqa

        positive_item = random.choice(positive_indices).item()
        positive = self._get(positive_item)

        # Compute the indices of all items that have a different label than the anchor
        negative_indices = argwhere(self.y != anchor.y)  # noqa
        negative_item = random.choice(negative_indices).item()
        negative = self._get(negative_item)

        return anchor, positive, negative

    def _get(self, item):
        # Return a named tuple with the requested data point
        return Signature(self.x[item], *[getattr(self, attr)[item] for attr in self.SINGULAR_ATTRIBUTES])

    def __len__(self):
        return len(self.y)

    def save(self, save_path):
        if not exists(save_path.parent):
            makedirs(save_path.parent)
        save({'x': self.x, **{attr: getattr(self, attr) for attr in self.SINGULAR_ATTRIBUTES}}, save_path)

    def clone(self, index):
        clone = KIDataset.__new__(KIDataset)
        for attr, value in self.__dict__.items():
            # Only index attribute if it's a tensor
            if isinstance(value, Tensor):
                value = value.clone()[index]
            # Otherwise, just take the same value (e.g. bool=True) OBS this will break with referenced values
            setattr(clone, attr, value)
        return clone

    def exclude_data(self, exclude):
        # Bool mask of what to exclude
        exclude_items = zeros(self.y.size(), dtype=torch_bool)
        # Take union with bool mask for every partition to exclude
        if 'PDON' in exclude:
            exclude_items = logical_or(exclude_items, self.y == KI_LABELS['PDON'])

        if 'pro' in exclude:
            exclude_items = logical_or(exclude_items, self.s == KI_SACCADE['pro'])

        if 'anti' in exclude:
            exclude_items = logical_or(exclude_items, self.s == KI_SACCADE['anti'])

        if 'vert' in exclude:
            exclude_items = logical_or(exclude_items, self.a == KI_AXIS['vert'])

        # For every attribute that is tensor, only keep the inverse of the excluded items
        for attr, value in self.__dict__.items():
            if isinstance(value, Tensor):
                setattr(self, attr, value[~exclude_items])

    # Flattens x to extract the individual segments from experiments.
    def flatten(self):
        # Save the original shape of x
        original_shape = self.x.shape
        self.x = self.x.view(-1, self.x.shape[-2], self.x.shape[-1])

        # Expand singular attributes to match the new shape of x
        for attr in self.SINGULAR_ATTRIBUTES:
            setattr(self, attr, getattr(self, attr).repeat_interleave(original_shape[1], dim=0))

    # Batches the segments into experiments according to the r attribute.
    def batch(self):
        sorting_indices = argsort(self.r)
        # Compute the number of segments in each experiment.
        batch_sizes = bincount(self.r)
        # Since the segments are split train/val, there will be some empty experiments. Remove them.
        batch_sizes = batch_sizes[batch_sizes > 0]

        # Sort the segments by experiment, batch them, and stack them into a tensor
        self.x = stack(self.x[sorting_indices].split(batch_sizes.tolist()))
        # Only keep one entry per experiment for the other attributes
        for attr in self.SINGULAR_ATTRIBUTES:
            entries_per_experiment = getattr(self, attr)[sorting_indices].split(batch_sizes.tolist())
            setattr(self, attr, stack([entries[0] for entries in entries_per_experiment]))

    @staticmethod
    def format_filename(train, bundle_as_trials, sources):
        return f"ki-{','.join(sources)}-{'trial' if bundle_as_trials else 'seg'}-{'train' if train else 'test'}.pth"


def populate_ki_trials(segmented_files, filenames):
    datapoints = []
    for trial, (trials, filename) in enumerate(zip(segmented_files, filenames)):
        individual, group, axis, saccade = KI_FILENAME_REGEX.findall(filename)[0]
        for t in trials:
            datapoints.append(Signature(
                x=t.values,
                y=KI_LABELS[group],
                z=int(individual),
                r=trial,
                a=KI_AXIS[axis],
                s=KI_SACCADE[saccade],
                g=KI_LABELS[group],
            ))
    return zip(*datapoints)


def populate_ki_sessions(segmented_files, filenames):
    datapoints = []
    for session, (segments, filename) in enumerate(zip(segmented_files, filenames)):
        individual, group, axis, saccade = KI_FILENAME_REGEX.findall(filename)[0]
        datapoints.append(Signature(
            x=[s.values for s in segments],
            y=KI_LABELS[group],
            z=int(individual),
            r=session,
            a=KI_AXIS[axis],
            s=KI_SACCADE[saccade],
            g=KI_LABELS[group],
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


def test():
    with open(f'{config_path}/leif.yaml', 'r') as reader:
        config = load_yaml(reader, Loader=FullLoader)

    processor = Leif(config)

    ds = KIDataset(data_processor=processor, train=True, bundle_as_sessions=False, use_triplets=False,
                   exclude=['vert'])
    binarize(ds)

    list_of_arrays = [t.x[0].numpy() for t in ds if t.y == 1]

    avg_power_spectrum, center_freq = compute_average_power_spectrum(list_of_arrays, SAMPLE_RATE, 30)

    plot_power_spectrum(center_freq, avg_power_spectrum)

    example = ds[0]

    plot_trial(example)
    ds.batch()
    ds.flatten()


if __name__ == '__main__':
    test()
