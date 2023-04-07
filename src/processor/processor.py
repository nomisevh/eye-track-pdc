"""
This file hosts our data preprocessing pipeline including sanitation, normalization, segmentation.
"""
from typing import List, Tuple

import numpy as np
from numpy import logical_and, ndarray, array
from pandas import DataFrame
from torch import inf
from tqdm import tqdm

from processor.interface import MainProcessor, BatchProcessor, Scissor
from utils.data import ZScoreMask
from utils.ki import SAMPLE_RATE, INVISIBLE_TARGET_DURATION


class Leif(MainProcessor):
    """
    Our main data preprocessor.
    """

    def __init__(self, config: dict):
        self.sanitizer = FileFilter(**config['sanitation'])
        normalization_method = config['file_normalization'].pop('which')
        if normalization_method == 'saccade_amplitude':
            self.file_normalizer = SaccadeAmplitudeNormalizer(**config['file_normalization'])
        elif normalization_method == 'target':
            self.file_normalizer = TargetBasedNormalizer(**config['file_normalization'])
        else:
            raise NotImplementedError(f'Normalization method {normalization_method} not implemented.')

        self.scissor = FocusScissor(sample_rate=SAMPLE_RATE, invisible_target_duration=INVISIBLE_TARGET_DURATION,
                                    **config['segmentation'])
        self.channels = config['channels']

    def __call__(self, frames: List[DataFrame], **kwargs) -> Tuple[List[List[DataFrame]], ndarray]:
        # Sanitize files
        frames = self.sanitizer(frames)

        # Normalize files
        frames = self.file_normalizer(frames)

        # For each file/trial:
        trials, min_segment_length = [], inf
        for frame in tqdm(frames, desc='segmenting time series', unit='ts'):
            # Segment
            segmented_frame = self.scissor(frame)

            # For every segment:
            for i, segment in enumerate(segmented_frame):
                # Normalize segment
                segment = normalize(segment, second_moment=False)
                # Compute velocity for every segment
                segment = compute_velocity(segment)
                # Select channels
                segmented_frame[i] = segment[self.channels]

            # Update minimum segment length
            min_segment_length = min(min_segment_length, min(map(lambda s: s.shape[0], segmented_frame)))
            # Append to segment list for file
            trials.append(segmented_frame)

        # TODO the min segment length might have to be synced between train and test
        # Trim segments to ensure equal length. Trim from the start of the segment since that is the noisiest part.
        for i, trial in enumerate(trials):
            for j, segment in enumerate(trial):
                trials[i][j] = segment[-min_segment_length:]

        return trials, self.sanitizer.mask


class SaccadeAmplitudeNormalizer(BatchProcessor):
    """
    Normalizes files individually based on the approximate median amplitude of all positive saccades.
    """
    NORMALIZE_CHANNELS = ["position", "drift", "target", "position_diff", "drift_diff"]

    def __init__(self, *, trailing_window_width, scaling_factor=10):
        """
        :param trailing_window_width: The width of the trailing window over which the median values are computed.
        :param scaling_factor: The factor by which we scale the saccade amplitude before normalizing.
            10 gives a good range of values for KI dataset.
        """
        self.n = trailing_window_width
        self.scaling_factor = scaling_factor

    def __call__(self, frames: List[DataFrame], **kwargs) -> List[DataFrame]:
        skipped = 0

        normalized_frames = []
        for df in frames:
            target = df["target"]
            target_diff = target.diff().fillna(0)
            anchors = [*target[target_diff != 0].index.tolist()]
            if not len(anchors):
                skipped += 1
                continue

            saccade_diffs = []
            for j in range(0, len(anchors) - 1, 2):
                # Use only positive values, as negative and positive saccades have different magnitude in eye position.
                if target_diff[anchors[j]] < 0:
                    continue

                saccade_start, saccade_end = anchors[j] - 1, anchors[j + 1]
                peak_value = np.median(df["position"][saccade_end - self.n:saccade_end])
                idle_value = np.median(df["position"][saccade_start - self.n:saccade_start])
                saccade_diffs.append(abs(peak_value - idle_value))
            if not len(saccade_diffs):
                raise Exception("no positive saccades found")

            df[self.NORMALIZE_CHANNELS] /= np.median(saccade_diffs)
            # scale the values to get a nice range
            df[self.NORMALIZE_CHANNELS] *= self.scaling_factor

            normalized_frames.append(df)

        if skipped > 0:
            print(f"skipped {skipped} files (no target movement)")
        return normalized_frames


class TargetBasedNormalizer(BatchProcessor):
    """
    Normalizes files based on the absolute maximum value of the target channel.
    """
    NORMALIZE_CHANNELS = ["position", "drift", "target", "position_diff", "drift_diff"]

    def __init__(self, *, scaling_factor=10):
        """
        :param scaling_factor: The factor by which we scale the saccade amplitude before normalizing.
            10 gives a good range of values for KI dataset.
        """
        self.scaling_factor = scaling_factor

    def __call__(self, frames: List[DataFrame], **kwargs) -> List[DataFrame]:
        # TODO I hope it works to do this in place
        for df in frames:
            df[self.NORMALIZE_CHANNELS] /= df["target"].abs().max()
            df[self.NORMALIZE_CHANNELS] *= self.scaling_factor
        return frames


class FileFilter(BatchProcessor):
    """
    Filters files based on z scores on a number of heuristics.
    """
    SERIES_HEADERS = ['position', 'drift']

    def __init__(self, *, position_thresholds, drift_thresholds):
        self.position_thresholds = position_thresholds.values()
        self.drift_thresholds = drift_thresholds.values()
        self.mask = None

    def __call__(self, frames: List[DataFrame], **kwargs) -> List[DataFrame]:
        """
        Filter the frames by max value, signal-to-noise ratio and mean velocity. The filter is applied with respect to
        the position and the drift independently.

        :param frames: A list holding the dataframes to be filtered
        """
        iterator = zip(FileFilter.SERIES_HEADERS, [self.position_thresholds, self.drift_thresholds])
        keep = np.ones(len(frames), dtype=bool)
        for header, thresholds in iterator:
            series = [df[header] for df in frames]
            heuristics = {
                'max value': [s.abs().max() for s in series],
                'mean velocity': [s.diff().fillna(0).abs().mean(axis=0) for s in series]
            }

            for (h_name, h_values), threshold in zip(heuristics.items(), thresholds):
                # Filter all remaining frames based on the z scores over the computed heuristic
                outlier = ZScoreMask(threshold)(h_values)
                keep = logical_and(keep, ~outlier)
                print(f'{outlier.sum()} files are {header} {h_name} outliers')

        # Filter the frames
        frames = array(frames, dtype=object)[keep]
        self.mask = keep
        print(f'skipped {(~keep).sum()} files in total')

        return frames


class FocusScissor(Scissor):
    """
    Segments a file such that the segments capture the periods where the individual is focusing in between saccades.
    """

    def __init__(self, *, sample_rate, post_saccade_time_threshold, exclude_first, invisible_target_duration,
                 invisible_target_mode=''):
        self.sample_rate = sample_rate
        self.post_saccade_threshold = int(sample_rate * post_saccade_time_threshold)
        self.exclude_first = exclude_first
        self.invisible_target_samples = int(invisible_target_duration * sample_rate)
        # Can take values 'exclude' or 'only'
        self.invisible_target_mode = invisible_target_mode

    def __call__(self, frame: DataFrame, **kwargs) -> List[DataFrame]:
        target = frame["target"]
        anchors = [0, *target[target.diff().fillna(0) != 0].index.tolist()]
        segments = []
        for i in range(0, len(anchors) - 1, 2):
            if i == 0 and self.exclude_first:
                continue

            s = frame.iloc[anchors[i] + self.post_saccade_threshold:anchors[i + 1], :].copy()

            if self.invisible_target_mode == 'exclude':
                s = self.exclude_invisible_target(s)
            elif self.invisible_target_mode == 'only':
                s = self.only_use_visible_target(s)

            # Add the initial timestamp of the segment
            start_time = s["Time (ms)"].iloc[0]
            s["Time (ms)"] -= start_time
            s.reset_index(drop=True, inplace=True)
            # Place the segment in the returning list
            segments.append(s)

        return segments

    def exclude_invisible_target(self, segment: DataFrame):
        return segment.iloc[:-self.invisible_target_samples, :].copy()

    def only_use_visible_target(self, segment: DataFrame):
        return segment.iloc[-self.invisible_target_samples:, :].copy()


def compute_velocity(df: DataFrame) -> DataFrame:
    """
    Computes the first-order differences of the position and drift w.r.t. time.
    Returns the same DataFrame with the velocity fields added.
    """
    d_time = df["Time (ms)"].diff().replace(0, np.nan).fillna(1)

    d_pos = df["position"].diff().fillna(0)
    d_drift = df["drift"].diff().fillna(0)

    d_pos_diff = df["position_diff"].diff().fillna(0)
    d_drift_diff = df["drift_diff"].diff().fillna(0)

    df["position_velocity"] = d_pos / d_time
    df["drift_velocity"] = d_drift / d_time

    df["position_diff_velocity"] = d_pos_diff / d_time
    df["drift_diff_velocity"] = d_drift_diff / d_time

    df['velocity_magnitude'] = np.linalg.norm(df[["position_velocity", "drift_velocity"]], axis=1)
    df['velocity_diff_magnitude'] = np.linalg.norm(df[["position_diff_velocity", "drift_diff_velocity"]],
                                                   axis=1)

    # Skip first data point as we lack info about velocity for it
    return df[1:]


# TODO maybe diffs shouldn't centered, as their magnitude and sign might be important
def normalize(df: DataFrame, second_moment: bool) -> DataFrame:
    df_cols = df[["position", "drift", "position_diff", "drift_diff"]]
    df[["position", "drift", "position_diff", "drift_diff"]] -= df_cols.mean(axis=0)
    if second_moment:
        df[["position", "drift", "position_diff", "drift_diff"]] /= df_cols.std(axis=0)
    return df
