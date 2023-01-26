from collections import namedtuple
from pathlib import Path

from torch import tensor, int as torch_int, save
from torch.utils.data import Dataset

from processor import DataProcessor
from utils.ki import LABELS as KI_LABELS, FILENAME_REGEX as KI_FILENAME_REGEX, AXIS as KI_AXIS, SACCADE as KI_SACCADE


class KIDataset(Dataset):
    Signature = namedtuple('Signature', ['x', 'y', 'z', 'r', 'a', 's'])

    def __init__(self, *, data_dir: Path, save_dir: Path, data_processor: DataProcessor, train: bool):
        # Do KI specific stuff: train/test files --> dataframes, filenames
        dataframes, filenames = ..., ...

        segmented_files = data_processor(dataframes)

        x, y, z, r, a, s = populate_ki(segmented_files, filenames)

        # Tensor with shape (N, M, T) holding the multivariate time series.
        # N is number of data points, M is the dimensionality and T is the length of the series.
        self.x = tensor(x).float().permute(0, 2, 1)
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

        # Pickle data for faster future loading TODO not sure if Pathlib Path works here, maybe need to resolve it
        save_filename = f"ki-dataset-{'train' if train else 'test'}"
        save({'x': self.x, 'y': self.y, 'z': self.z, 'r': self.r, 'a': self.a, 's': self.s},
             f'{save_dir}/{save_filename}')

    def __getitem__(self, item):
        return self.Signature(self.x[item], self.y[item], self.z[item], self.r[item], self.a[item], self.s[item])

    def __len__(self):
        return len(self.y)


def populate_ki(segmented_files, filenames):
    datapoints = []
    for trial, (segments, filename) in enumerate(zip(segmented_files, filenames)):
        individual, group, axis, saccade = KI_FILENAME_REGEX.findall(filename)[0]
        for seg in segments:
            datapoints.append(KIDataset.Signature(
                x=seg,
                y=KI_LABELS[group],
                z=int(individual),
                r=trial,
                a=KI_AXIS[axis],
                s=KI_SACCADE[saccade],
            ))
    return zip(*datapoints)
