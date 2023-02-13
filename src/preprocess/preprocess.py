import abc
import copy
import dataclasses
import multiprocessing as mp
import os
from itertools import chain
from typing import List, Any, Dict, Callable

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from preprocess.ifaces import MTSData, Operation, AtomicOperation, Capturable, Scissor, BatchOperation
from utils.path import config_path


# ---------------------------------------------------------------


class TrialSegment(MTSData):
    frame_start: int or torch.Tensor
    frame_end: int or torch.Tensor

    @property
    def __len__(self):
        return self.frame_end - self.frame_start + 1

    def to_torch_(self) -> None:
        self.x = torch.from_numpy(self.x.values).float()
        self.frame_start = torch.tensor(self.frame_start).int(),
        self.frame_end = torch.tensor(self.frame_end).int()

    def trim_(self, new_len: int):
        self.x = self.x[-new_len:]
        self.frame_start = self.frame_end - new_len + 1

    def state_dict(self) -> Dict[str, torch.Tensor or dict]:
        if type(self.x) != torch.Tensor:
            self.to_torch_()
        return {
            'x': self.x,
            'frame_start': self.frame_start,
            'frame_end': self.frame_end,
            'removed': self.removed,
        }

    def load_state_dict(self, sd: Dict[str, torch.Tensor or dict]) -> None:
        self.x = sd['x']
        self.frame_start = sd['frame_start']
        self.frame_end = sd['frame_end']
        self.removed = sd['removed']


class Trial(MTSData):
    y: int or torch.Tensor
    z: int or torch.Tensor
    meta: Dict[str, Any] = dataclasses.field(default_factory=lambda: {})
    segments: List[TrialSegment] = dataclasses.field(default_factory=lambda: [])

    def __init__(self, mts: pd.DataFrame, label: int, person_id: int, **metadata):
        self.x = mts
        self.y = int(label)
        self.z = int(person_id)
        self.meta = metadata
        self.segments = []

    @property
    def __len__(self):
        return self.segments.__len__()

    @property
    def usable_segments(self):
        return [s for s in self.segments if not s.removed]

    def to_torch_(self) -> None:
        self.x = torch.from_numpy(self.x.values).float()
        self.y = torch.tensor(self.y).int()
        self.z = torch.tensor(self.z).int()
        for s in self.segments:
            s.to_torch_()

    def trim_segments_(self, new_len: int or None = None):
        if new_len is None:
            new_len = min(len(s.x) for s in self.segments)
        for segment in self.segments:
            segment.trim_(new_len)

    def state_dict(self) -> Dict[str, torch.Tensor or dict]:
        if type(self.x) != torch.Tensor:
            self.to_torch_()
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'meta': self.meta,
            'segments': [s.state_dict() for s in self.segments],
            'removed': self.removed,
        }

    def load_state_dict(self, sd: Dict[str, torch.Tensor or dict]) -> None:
        self.x = sd['x']
        self.y = sd['y']
        self.z = sd['z']
        self.meta = sd['meta']
        self.removed = sd['removed']
        self.segments = []
        for sd_s in sd['segments']:
            s = TrialSegment(x=pd.DataFrame([]))
            s.load_state_dict(sd_s)
            self.segments.append(s)

    @staticmethod
    def empty() -> 'Trial':
        return Trial(mts=pd.DataFrame([]), label=-1, person_id=-1, empty=True)


# ---------------------------------------------------------------


class SelectChannels(AtomicOperation, Capturable):
    def __init__(self, channels: List[str]):
        self.col_names = list(channels)

    def __call__(self, mts: MTSData) -> MTSData:
        mts.only_cols_(self.col_names)
        return mts

    def state_dict(self) -> Dict[str, Any]:
        return dict(col_names=self.col_names)

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.col_names = sd['col_names']


class RenameChannels(AtomicOperation, Capturable):
    # Case-insensitive mapping of {old_name1: new_name1, ..., old_nameM: new_nameM}
    NAME_MAP = {
        'Time (ms)': 'time',
        'x': 'position',
        'x_pos': 'position',
        'y': 'drift',
        'y_pos': 'drift',
    }

    def __call__(self, mts: MTSData) -> MTSData:
        mts.x.rename(columns=self.NAME_MAP, inplace=True)
        return mts

    def state_dict(self) -> Dict[str, Any]:
        return dict()

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        ...


class ComputeVelocity(AtomicOperation, Capturable):
    """
    ComputeVelocity Class:
    Computes the first-order differences of x_pos and y_pos w.r.t. time. Returns the same DataFrame with the velocity
    fields added.
    """

    def __init__(self, include_magnitude: bool = False):
        """
        ComputeVelocityTransform class constructor.

        Parameters
        ----------
        include_magnitude: bool
                           Set to True to return the velocity magnitude alongside velocities
        """
        self.include_magnitude = include_magnitude

    def __call__(self, mts: MTSData) -> MTSData:
        df = mts.x
        d_time = df['time'].diff().replace(0, np.nan).fillna(1)

        d_pos = df['position'].diff().fillna(0)
        d_drift = df['drift'].diff().fillna(0)

        d_pos_diff = df['position_diff'].diff().fillna(0)
        d_drift_diff = df['drift_diff'].diff().fillna(0)

        df['position_velocity'] = d_pos / d_time
        df['drift_velocity'] = d_drift / d_time

        df['position_diff_velocity'] = d_pos_diff / d_time
        df['drift_diff_velocity'] = d_drift_diff / d_time

        if self.include_magnitude:
            df['velocity_magnitude'] = np.linalg.norm(df[['position_velocity', 'drift_velocity']], axis=1)
            df['velocity_diff_magnitude'] = np.linalg.norm(df[['position_diff_velocity', 'drift_diff_velocity']],
                                                           axis=1)
        # Skip first data point as we lack info about velocity for it
        mts.x = mts.x[1:]
        return mts

    def state_dict(self) -> Dict[str, Any]:
        return {'include_magnitude': int(self.include_magnitude)}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.include_magnitude = bool(sd['include_magnitude'])


class NormalizeSaccadeAmplitude(AtomicOperation, Capturable):
    """
    NormalizeSaccadeAmplitude Class:
        1. Normalize files individually based on the (approximate) median amplitude of all positive saccades.
        2. When there is no target movement mark trial files as removed
    """

    def __init__(self,
                 lookup_samples: int = 10,
                 scaling_factor: float = 0.5,
                 col_names: List[str] or None = ('position', 'drift', 'target')):
        """
        NormalizeSaccadeAmplitude class constructor.

        Parameters
        ----------
        lookup_samples: int
                        Number of samples to take into account (i.e. steps after the target ON to consider for the low
                        value and steps before target "OFF" to consider as the high one)
        col_names: list of str
                   The columns to be normalized.
        """
        self.lookup_samples = lookup_samples
        self.scaling_factor = scaling_factor
        self.col_names = list(col_names)

    def __call__(self, trial: Trial) -> Trial:
        target = trial.x['target']
        target_diff = target.diff().fillna(0)
        anchors = [*target[target_diff != 0].index.tolist()]
        if not len(anchors):
            trial.removed = True
            return trial

        saccade_diffs = {col: [] for col in self.col_names}
        for i in range(0, len(anchors) - 1, 2):
            # Use the first positive value, as negative and positive saccades have different magnitude in eye position.
            if target_diff[anchors[i]] > 0:
                for col in self.col_names:
                    saccade = trial.x[col].iloc[anchors[i] - 1: anchors[i + 1]]
                    high_value = np.median(saccade.iloc[-self.lookup_samples:])
                    low_value = np.median(saccade.iloc[:self.lookup_samples])
                    saccade_diffs[col].append(abs(high_value - low_value))

        if not len(saccade_diffs):
            trial.removed = True
            return trial

        # Normalize by one tenth of the reference value to get a nice range
        for col in self.col_names:
            trial.x[col] /= (np.median(saccade_diffs[col]) * self.scaling_factor)
        return trial

    def state_dict(self) -> Dict[str, Any]:
        return dict(lookup_samples=self.lookup_samples, scaling_factor=self.scaling_factor, col_names=self.col_names)

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.lookup_samples = sd['lookup_samples']
        self.scaling_factor = sd['scaling_factor']
        self.col_names = sd['col_names']


class ZScoreFilter(BatchOperation, Capturable):
    def __init__(self, heuristic: Callable, threshold: float or str, col_name: str):
        self.heuristic = heuristic
        self.threshold = threshold
        self.col_name = col_name
        self.state: Dict[str, np.ndarray or None] = {
            'values_mean': None,
            'values_std': None,
            'values_thres': None,
            'values_zscores': None,
        }

    def __call__(self, mts_list: List[MTSData]) -> List[MTSData]:
        #   - get values
        values = np.array([self.heuristic(mts.x[self.col_name]) for mts in mts_list])
        #   - compute stats
        if self.state['values_mean'] is None or self.state['values_std'] is None:
            values_mean = values.mean(axis=0)
            self.state['values_mean'] = values_mean
            values_std = values.std(axis=0)
            self.state['values_std'] = values_std
        #   - compute z-scores
        values_zscores = np.abs((values - self.state['values_mean']) / self.state['values_std'])
        #   - compute threshold
        if self.state['values_thres'] is None:
            if type(self.threshold) == str and self.threshold.startswith('auto'):
                try:
                    n_stds = float(self.threshold.split(':')[-1])
                    assert 0 < n_stds < 1e10
                except ValueError or AssertionError:
                    n_stds = 2.94
                values_thres = values_zscores.mean(axis=0) + n_stds * values_zscores.std(axis=0)
                # print('values_thres', values_thres)
            elif type(self.threshold) == float:
                values_thres = self.threshold * np.ones_like(self.state['values_mean'])
            elif type(self.threshold) == list:
                values_thres = np.array(self.threshold)
            else:
                raise AttributeError(f'thres has wrong type: {type(self.threshold)}')
            self.state['values_thres'] = values_thres
        #   - update attributes of trials to be removed
        remove = np.greater(values_zscores, self.state['values_thres'])
        # for col, max_thres_col in enumerate(self.state['values_thres'].tolist()):
        #     remove = np.logical_or(remove, np.greater(values_zscores[:, col], max_thres_col))
        for mts, remove_i in zip(mts_list, list(remove.tolist())):
            mts.removed = remove_i
        return mts_list

    def state_dict(self) -> Dict[str, Any]:
        return dict(col_name=self.col_name, state=copy.deepcopy(self.state))

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.col_name = sd['col_name']
        self.state = copy.deepcopy(sd['state'])


class RemoveNoisyData(BatchOperation, Capturable, metaclass=abc.ABCMeta):
    """
    RemoveNoisyData Class:
    This class is used to remove noisy data. Those can either be trial files or single segments.
    """

    def __init__(self, heuristics: Dict[str, Callable], col_thresholds: Dict[str, Dict[str, float or str]]):
        self.filters: Dict[str, Dict[str, ZScoreFilter]] = {
            col: {
                hk: ZScoreFilter(h, thresholds[hk], col_name=col)
                for hk, h in heuristics.items()
            }
            for col, thresholds in col_thresholds.items()
        }

    def state_dict(self) -> Dict[str, Dict[str, Dict[str, dict]]]:
        return dict(filters={
            col: {
                hk: hfilter.state_dict()
                for hk, hfilter in col_filters.items()
            }
            for col, col_filters in self.filters.items()
        })

    def load_state_dict(self, sd: Dict[str, Dict[str, Dict[str, dict]]]) -> None:
        for col, col_filters_sds in sd['filters'].items():
            for hk, hfilter_sd in col_filters_sds.items():
                self.filters[col][hk].load_state_dict(hfilter_sd)


class RemoveNoisyTrials(RemoveNoisyData):
    """
    RemoveNoisyTrials Class:
    This class is used to remove noisy files entirely. So trial files will either pass through or get rejected.
    """

    def __call__(self, trials: List[Trial]) -> List[Trial]:
        for col, col_filters in self.filters.items():
            for hk, hfilter in col_filters.items():
                trials = hfilter.__call__(mts_list=[t for t in trials if not t.removed])
        return trials


class RemoveNoisySegments(RemoveNoisyData):
    """
    RemoveNoisySegments Class:
    This class is used to remove noisy segments.
    """

    def __call__(self, trials: List[Trial]) -> List[Trial]:
        for col, col_filters in self.filters.items():
            for hk, hfilter in col_filters.items():
                hfilter.__call__(mts_list=self.__class__.gather_segments([t for t in trials if not t.removed]))
        return trials

    @staticmethod
    def gather_segments(trials: List[Trial]) -> List[TrialSegment]:
        return list(chain(*[t.usable_segments for t in trials]))


# ---------------------------------------------------------------

class FixationScissor(Scissor):
    """
    FixationScissor Class:
    Segments a file such that the segments capture the periods where the individual is focusing in between saccades.
    """

    def __init__(self, *,
                 sample_rate: int,
                 post_saccade_time_threshold: float,
                 exclude_first: bool = True,
                 target_col: str = 'target'):
        self.sample_rate = sample_rate
        self.d = post_saccade_time_threshold
        self.exclude_first = exclude_first
        self.target_col = target_col

    def __call__(self, trial: Trial, **kwargs) -> Trial:
        th = int(self.sample_rate * self.d)
        target = trial.x[self.target_col]
        anchors = [0, *target[target.diff().fillna(0) != 0].index.tolist()]
        trial.segments = []
        for i in range(0, len(anchors) - 1, 2):
            s = trial.x.iloc[anchors[i] + th:anchors[i + 1], :].copy()
            # Add the initial timestamp of the segment
            start_time = s['time'].iloc[0]
            s['time'] -= start_time
            s.reset_index(drop=True, inplace=True)
            # Place the segment in the trial list
            segment = TrialSegment(s)
            segment.frame_start = start_time
            segment.frame_end = start_time + len(s['position']) - 1
            trial.segments.append(segment)
        return trial

    def state_dict(self) -> Dict[str, Any]:
        return dict(sample_rate=self.sample_rate, d=self.d, exclude_first=self.exclude_first)

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.sample_rate = sd['sample_rate']
        self.d = sd['d']
        self.exclude_first = sd['exclude_first']


class ExtractSegments(AtomicOperation, Capturable):
    def __init__(self, scissor: Scissor):
        self.scissor = scissor

    def __call__(self, trial: Trial) -> Trial:
        return self.scissor(trial)

    def state_dict(self) -> Dict[str, Any]:
        return dict(scissor=self.scissor.state_dict())

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        return self.scissor.load_state_dict(sd['scissor'])


class TrimTrialSegments(AtomicOperation, Capturable):
    """
    TrimTrialSegments Class:
    Class to trim all the segments inside each trial. Trim length can be derived either per trial or through all trials.
    """

    def __init__(self, target_length: float or None = None, per_trial: bool = False):
        self.target_length = target_length
        self.per_trial = per_trial

    def __call__(self, trials: List[Trial] or Trial) -> List[Trial] or Trial:
        if type(trials) == list:
            if self.per_trial:
                for trial in trials:
                    trial.trim_segments_(new_len=None)
            else:
                if self.target_length is None:
                    segments = RemoveNoisySegments.gather_segments(trials)
                    self.target_length = min(len(s.x) for s in segments)
                for trial in trials:
                    trial.trim_segments_(new_len=self.target_length)
        else:

            trials.trim_segments_(new_len=None)
        return trials

    def state_dict(self) -> Dict[str, Any]:
        return dict(target_length=self.target_length, per_trial=self.per_trial)

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.target_length = sd.get('target_length', None)
        self.per_trial = sd.get('per_trial', False)


# ---------------------------------------------------------------

class SequentialProcessor(AtomicOperation, Capturable):
    def __init__(self, ops: List[Operation and Capturable]):
        self.ops = ops

    def __call__(self, trial: Trial) -> Trial:
        for op in self.ops:
            trial = op(trial)
        return trial

    def state_dict(self) -> Dict[str, Any]:
        return dict(ops=[
            op.state_dict()
            for op in self.ops
        ])

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        assert 'ops' in sd.keys() and len(sd['ops']) == len(self.ops)
        for op_i, op in enumerate(self.ops):
            op.load_state_dict(sd['ops'][op_i])


class ParallelFileProcessor(BatchOperation, Capturable):
    """
    ParallelFileProcessor Class:
    Parallel executor for sequential operations on trial files. The files
    """

    def __init__(self, map_ops: List[Operation and Capturable], n_process: int = mp.cpu_count()):
        for op in map_ops:
            assert isinstance(op, Operation) and isinstance(op, Capturable)
        self.atomic_op = SequentialProcessor(map_ops)
        self.n_process = n_process

    def __call__(self, trials: List[Trial]) -> List[Trial]:
        with mp.Pool(processes=self.n_process) as pool:
            trials = pool.map(self.atomic_op, [t for t in trials if not t.removed])
        return trials

    def state_dict(self) -> Dict[str, Any]:
        return dict(atomic_op=self.atomic_op.state_dict(), n_process=self.n_process)

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.atomic_op.load_state_dict(sd['atomic_op'])
        self.n_process = sd['n_process']


class CompositeProcessor(SequentialProcessor):
    """
    CompositeProcessor Class:
    This class defines our entire data pre-processing pipeline through a sequence of operations.
    """

    def __init__(self, config: Dict[str, Any], trim_per_trial: bool = True, verbose: bool = True):
        ops = [
            ParallelFileProcessor(
                map_ops=[
                    RenameChannels(),
                    ComputeVelocity(**config['compute_velocity']),
                    SelectChannels(channels=config['channels']),  # todo: update config accordingly
                    NormalizeSaccadeAmplitude(**config['normalization']['saccade_amplitude']),
                ]
            ),
            RemoveNoisyTrials(
                heuristics={
                    'max_value': lambda s: s.abs().max(),
                    'snr': lambda s: s.abs().max() / (s.var() + np.finfo(float).eps),
                    'mean_vel': lambda s: s.diff().fillna(0).abs().mean(axis=0),
                },
                col_thresholds=config['sanitation']['trial']['thresholds'],
            ),
            ParallelFileProcessor(
                map_ops=[
                    ExtractSegments(scissor=FixationScissor(**config['segmentation'])),
                ]
            ),
            RemoveNoisySegments(
                heuristics={
                    'max_value': lambda s: s.abs().max(),
                },
                col_thresholds=config['sanitation']['segment']['thresholds'],
            ),
            ParallelFileProcessor(
                map_ops=[
                    TrimTrialSegments(per_trial=True)
                ]
            ) if trim_per_trial else TrimTrialSegments(per_trial=False)
        ]
        super().__init__(ops=ops)
        self.n_files = None
        self.n_segments = None
        self.verbose = verbose

    def __call__(self, trials: List[Trial]) -> List[Trial]:
        # self.n_files, self.n_segments = [len(trials)], [sum(len(t.segments) for t in trials if not t.removed)]
        pbar = tqdm(self.ops)
        for op in pbar:
            pbar.set_description(op.__class__.__name__)
            trials = op([t for t in trials if not t.removed])
            # self.n_files.append(len([t for t in trials if not t.removed]))
            # self.n_segments.append(
            #     sum(len(t.usable_segments) for t in trials if not t.removed)
            # )
        # if self.verbose:
        #     import matplotlib.pyplot as plt
        #     fig, ax = plt.subplots()
        #     ax.plot(self.n_files, '-or', label='files')
        #     ax.set_ylabel('# files', color='red')
        #     ax2 = ax.twinx()
        #     self.n_segments[:3] = [self.n_segments[3]] * 3
        #     ax2.plot(self.n_segments, '-ob', label='segments')
        #     ax2.set_ylabel('# segments', color='blue')
        #     plt.title('How many data are removed?')
        #     plt.tight_layout()
        #     if not os.path.exists('pp_numbers.pdf'):
        #         plt.savefig('pp_numbers.pdf')
        #     else:
        #         plt.savefig('pp_numbers_second_time.pdf')
        #     plt.show()
        #     print(self.n_segments)
        #     print(self.n_files)
        return [t for t in trials if not t.removed]

    @staticmethod
    def from_config(config_name: str, **kwargs) -> 'CompositeProcessor':
        with open(os.path.join(str(config_path), f'{config_name}.yaml'), 'r') as config_fp:
            config = yaml.load(config_fp, Loader=yaml.FullLoader)
        return CompositeProcessor(config, **kwargs)
