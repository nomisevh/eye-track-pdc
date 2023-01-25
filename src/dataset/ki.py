import os
from collections import namedtuple
from itertools import chain
from pathlib import Path
from typing import Optional, List, Any, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset.ki_utils import load_ki_trials
from utils.path import data_path
from preprocess.base import Trial, CompositeProcessor


class KIDataset(Dataset):
    """
    KISegmentsDataset Class:
    This is used to access KI Eye-tracking data for PDC as a torch-compatible Dataset wrapper. The resulting dataset is
    formed by a 5-uple (x,y,z,r,a,s), where x is the MTS, y the label, z the individual, r the trial index, a the axis
    of the experiment and s the saccade type.
    """
    Signature = namedtuple('Signature', ['x', 'y', 'z', 'r', 'a', 's'])

    def __init__(self, *,
                 train: bool,
                 which: str = 'trials',
                 ki_data_dirname: str = 'ki',
                 config: str = 'ki_auto',
                 data_sources: List[str] = ('HC',),
                 save_dir: Optional[Path or str] = None,
                 verbose: bool = True):
        """
        KIDataset class constructor.

        Parameters
        ----------
        train         : bool
                        Set to true/false for the training/testing set.
        config        : str
                        The config to use for the pre-processing pipeline.
        data_sources  : list of str
                        List of data sources. Defaults to ("HC", "PD_ON", "PD_OFF").
        save_dir      : Path or str, optional
                        (Optional) Path to store processed data. Default to data directory if not provided.
        verbose       : bool
                        Set to true to enable logging of the resulting data distributions.
        which         : str
                        One of "segments", "trials". This is the quantity that is yielded at every `__getitem__` call.
        """
        self.data_path = os.path.join(str(data_path), ki_data_dirname)
        self.save_path = os.path.join(str(save_dir if save_dir is not None else data_path), ki_data_dirname)
        self.config = config
        self.which = which
        self.preprocessor = CompositeProcessor.from_config(config, trim_per_trial=which == 'trials')

        # Generate preprocessor checkpoint
        pp_sd_filename = self.get_sd_filename(data_sources=data_sources) + '.pth'
        pp_sd_filepath = os.path.join(self.save_path, pp_sd_filename)
        if not os.path.exists(pp_sd_filepath):
            # Stats file is not found: Need to generate this from training data
            print(f'\t[{self.__class__.__name__}][__init__] Generating preprocessing state dict...')
            #   - load all training trial files found in the source folder(s)
            trials = list(chain(*[
                load_ki_trials(os.path.join(self.data_path, 'train', src))
                for src in data_sources
            ]))
            #   - preprocess trial files
            self.preprocessor(trials)
            torch.save(self.preprocessor.state_dict(), pp_sd_filepath)
            print(f'\t[{self.__class__.__name__}][__init__] State dict saved at: {pp_sd_filepath}')

        # Generate (processed) data checkpoint
        ckpt_filename = self.get_ckpt_filename(data_sources=data_sources) + '.pth'
        ckpt_filepath = os.path.join(self.data_path, 'train' if train else 'test', ckpt_filename)
        if not os.path.exists(ckpt_filepath):
            print(f'\t[{self.__class__.__name__}][__init__] Generating data checkpoint...')
            # Load all trial files found in the source folder(s)
            trials = list(chain(*[
                load_ki_trials(os.path.join(self.data_path, 'train' if train else 'test', src))
                for src in data_sources
            ]))

            # Initialize the frozen preprocessor
            self.preprocessor.load_state_dict(torch.load(pp_sd_filepath))
            # Preprocess trial files
            processed_trials = self.preprocessor(trials)
            # Save processed trials
            torch.save([t.state_dict() for t in processed_trials], ckpt_filepath)

        # Load dataset
        trials = []
        for ti, t_sd in enumerate(torch.load(ckpt_filepath)):
            t = Trial.empty()
            t.load_state_dict(t_sd)
            if not t.removed:
                trials.append(t)
        self.x, self.y, self.z, self.r, self.a, self.s = self.load_from_trials(trials)

        if verbose:
            self.log_data_distribution()

    def get_sd_filename(self, data_sources: List[str]) -> str:
        return f'ki_pp_sd__{self.config}__{",".join(data_sources).lower().replace("_", "").replace(",", "_")}'

    def get_ckpt_filename(self, data_sources: List[str]) -> str:
        return f'ki_ckpt__{self.config}__{",".join(data_sources).lower().replace("_", "").replace(",", "_")}'

    def load_from_trials(self, trials: List[Trial]):
        # Create segments dataset
        x, y, z, r, a, s = self.__class__.populate_ki(trials, which=self.which)
        if self.which == 'segments':
            # Tensor with shape (N, M, T) holding the multivariate time series.
            # N is number of data points, M is the dimensionality and T is the length of the series.
            x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True).float()
        # Tensor with shape (N) holding the labels
        y = torch.tensor(y).float()
        # Tensor with shape (N) holding which patient the data points belong to
        z = torch.tensor(z, dtype=torch.int)
        # Tensor with shape (N) holding which trial each segment belongs to
        r = torch.tensor(r, dtype=torch.int)
        # Tensor with shape (N) holding the axis each segment is aligned with (0:'horiz', 1:'vert')
        a = torch.tensor(a, dtype=torch.int)
        # Tensor with shape (N) holding the saccade type of each segment (0:'pro', 1:'anti')
        s = torch.tensor(s, dtype=torch.int)

        return x, y, z, r, a, s

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index: Any) -> Tuple[float, int, int, int, int, int]:
        return self.x[index], self.y[index], self.z[index], self.r[index], self.a[index], self.s[index]

    # def get_cloned(self, index: Any) -> Tuple[float, int, int, int, int, int]:
    #     return self.x.clone()[index], self.y.clone()[index], self.z.clone()[index], self.t.clone()[index], \
    #         self.a.clone()[index], self.s.clone()[index]

    def log_data_distribution(self):
        tot = self.y.shape[0]
        n_pd_off = (self.y == 1).sum()
        n_pd_on = (self.y == 2).sum()
        n_hc = self.y.shape[0] - n_pd_on - n_pd_off
        print(f"[KIDataset] datapoint distribution over {tot} points:"
              f"\n\t\t\tPD_OFF: {n_pd_off / tot:.2%}"
              f"\n\t\t\tPD_ON: {n_pd_on / tot:.2%}"
              f"\n\t\t\tHC: {n_hc / tot:.2%}")

    @staticmethod
    def from_hc_pdoff(train: bool = True) -> 'KIDataset':
        return KIDataset(
            train=train,
            data_sources=['HC', 'PD_OFF'],
        )

    @staticmethod
    def populate_ki(processed_trials: List[Trial], which: str) -> Any:
        datapoints = []
        for trial_i, trial in enumerate(processed_trials):
            if which == 'segments':
                for segment in trial.usable_segments:
                    datapoints.append(KIDataset.Signature(
                        x=segment.x.values if isinstance(segment.x, pd.DataFrame) else segment.x,
                        y=trial.y,
                        z=trial.z,
                        r=trial_i,
                        a=trial.meta['axis'],
                        s=trial.meta['saccade'],
                    ))
            else:
                datapoints.append(KIDataset.Signature(
                    x=trial,
                    y=trial.y,
                    z=trial.z,
                    r=trial_i,
                    a=trial.meta['axis'],
                    s=trial.meta['saccade'],
                ))
        return zip(*datapoints)


if __name__ == '__main__':
    _ds = KIDataset(train=True, which='segments', ki_data_dirname='KI', data_sources=['HC', 'PD_OFF', 'PD_ON'])
    # _ds[0]
