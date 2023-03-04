"""
This file hosts our data preprocessing pipeline including sanitation, normalization, segmentation.
"""
from typing import List

import numpy as np
from numpy import finfo
from pandas import DataFrame
from torch import inf
from tqdm import tqdm

from processor.interface import MainProcessor, BatchProcessor, Scissor
from utils.data import ZScoreMask
from utils.ki import SAMPLE_RATE


class Leif(MainProcessor):
    """
    Our main data preprocessor.
    """

    def __init__(self, config: dict):
        self.sanitizer = FileFilter(**config['sanitation'])
        self.file_normalizer = SaccadeAmplitudeNormalizer(**config['file_normalization'])
        self.scissor = FocusScissor(sample_rate=SAMPLE_RATE, **config['segmentation'])
        self.channels = config['channels']

    def __call__(self, frames: List[DataFrame], train=True) -> List[List[DataFrame]]:
        # Sanitize files
        if train:
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

            # update minimum segment length
            min_segment_length = min(min_segment_length, min(map(lambda s: s.shape[0], segmented_frame)))
            # Append to segment list for file
            trials.append(segmented_frame)

        # Sanitize segments w.r.t all files
        # TODO the min segment length might have to be synced between train and test
        # Trim segments
        for i, trial in enumerate(trials):
            for j, segment in enumerate(trial):
                trials[i][j] = segment[-min_segment_length:]

        return trials


class SaccadeAmplitudeNormalizer(BatchProcessor):
    """
    Normalizes files individually based on the approximate median amplitude of all positive saccades.
    """
    NORMALIZE_CHANNELS = ["position", "drift", "target"]

    def __init__(self, *, trailing_window_width, scaling_factor=0.1):
        """
        :param trailing_window_width: The width of the trailing window over which the median values are computed.
        :param scaling_factor: The factor by which we scale the saccade amplitude before normalizing.
            0.1 gives a good range of values for KI dataset.
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

            # scale the reference value to get a nice range
            # TODO normalize diffs
            df[self.NORMALIZE_CHANNELS] /= (np.median(saccade_diffs) * self.scaling_factor)

            normalized_frames.append(df)

        if skipped > 0:
            print(f"skipped {skipped} files (no target movement)")
        return normalized_frames


class FileFilter(BatchProcessor):
    """
    Filters files based on z scores on a number of heuristics.
    """
    SERIES_HEADERS = ['position', 'drift']

    def __init__(self, *, position_thresholds, drift_thresholds):
        self.position_thresholds = position_thresholds.values()
        self.drift_thresholds = drift_thresholds.values()

    def __call__(self, frames: List[DataFrame], **kwargs) -> List[DataFrame]:
        """
        Filter the frames by max value, signal-to-noise ratio and mean velocity. The filter is applied with respect to
        the position and the drift independently.

        :param frames: A list holding the dataframes to be filtered
        """
        iterator = zip(FileFilter.SERIES_HEADERS, [self.position_thresholds, self.drift_thresholds])
        for header, thresholds in iterator:
            series = [df[header] for df in frames]
            heuristics = {
                'max value': [s.abs().max() for s in series],
                'snr': [s.abs().max() / (s.var() + finfo(float).eps) for s in series],
                'mean velocity': [s.diff().fillna(0).abs().mean(axis=0) for s in series]
            }

            for (h_name, h_values), threshold in zip(heuristics.items(), thresholds):
                z_mask = ZScoreMask(threshold)
                n_before = len(frames)
                # Filter all remaining frames based on the z scores over the computed heuristic
                frames = [frame for frame, outlier in zip(frames, z_mask(h_values)) if not outlier]
                print(f'skipped {abs(len(frames) - n_before)} files ({header} {h_name} outlier)')

        return frames


class FocusScissor(Scissor):
    """
    Segments a file such that the segments capture the periods where the individual is focusing in between saccades.
    """

    def __init__(self, *, sample_rate, post_saccade_time_threshold, exclude_first):
        self.sample_rate = sample_rate
        self.d = post_saccade_time_threshold
        self.exclude_first = exclude_first

    def __call__(self, frame: DataFrame, **kwargs) -> List[DataFrame]:
        th = int(self.sample_rate * self.d)

        target = frame["target"]
        anchors = [0, *target[target.diff().fillna(0) != 0].index.tolist()]
        segments = []
        min_seg_length = inf
        for i in range(0, len(anchors) - 1, 2):
            if i == 0 and self.exclude_first:
                continue

            s = frame.iloc[anchors[i] + th:anchors[i + 1], :].copy()
            # Add the initial timestamp of the segment
            start_time = s["Time (ms)"].iloc[0]
            s["Time (ms)"] -= start_time
            s.reset_index(drop=True, inplace=True)
            # Place the segment in the returning list
            segments.append(s)

        return segments


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
