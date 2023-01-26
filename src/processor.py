"""
This file hosts our data preprocessing pipeline including sanitation, normalization, segmentation.
"""
import abc
from abc import ABC
from typing import List

import numpy as np
from numpy import finfo
from pandas import DataFrame

from utils.data import ZScoreFilter
from utils.ki import SAMPLE_RATE


class DataProcessor(ABC):

    @abc.abstractmethod
    def __call__(self, frames: List[DataFrame], **kwargs) -> List[List[DataFrame]]:
        """
        Processes and segments a multivariate time series (MTS)

        :param frames: A list containing dataframes which hold the MTS
        :param kwargs: Any processor-specific arguments
        :return: A list holding all the processed and segmented MTS
        """
        raise NotImplementedError


# TODO File / Trial / something else?
class FileProcessor(ABC):

    @abc.abstractmethod
    def __call__(self, frames: List[DataFrame], **kwargs) -> List[DataFrame]:
        """
        Processes a list of multivariate time series (MTS)

        :param frames: A list containing dataframes which hold the MTS
        :param kwargs: Any processor-specific arguments
        :return: A list holding the processed MTS
        """
        raise NotImplementedError


class Scissor(ABC):

    @abc.abstractmethod
    def __call__(self, frame: DataFrame, **kwargs) -> List[DataFrame]:
        """
        Segments a multivariate time series (MTS)

        :param frame: A dataframe holding the MTS
        :param kwargs: Any processor-specific arguments
        :return: A list holding the segmented MTS
        """
        raise NotImplementedError


class SegmentProcessor(ABC):

    @abc.abstractmethod
    def __call__(self, frames: DataFrame, **kwargs) -> DataFrame:
        """
        Processes a segment of a multivariate time series (MTS)

        :param frames: A dataframes holding the MTS segment
        :param kwargs: Any processor-specific arguments
        :return: The processed MTS segment
        """
        raise NotImplementedError


class Leif(DataProcessor):
    """
    Our main data preprocessor.
    """

    def __init__(self, train: bool, config: dict):
        self.train = train
        self.sanitizer = FileFilter(**config['sanitation'])
        self.file_normalizer = SaccadeAmplitudeNormalizer(**config['file_normalization'])
        self.scissor = FocusScissor(sample_rate=SAMPLE_RATE, **config['segmentation'])

    def __call__(self, frames: List[DataFrame], **kwargs) -> List[DataFrame]:
        # TODO should we sanitize at all if test?
        # Sanitize files
        if self.train:
            frames = self.sanitizer(frames)

        # Normalize files
        frames = self.file_normalizer(frames)

        # For each file:
        # Segment
        for frame in frames:
            segmented_frame = self.scissor(frame)

            # For every segment:
            # Normalize segment
            # compute velocity
            # Record segment length TODO this might be able to simplify, e.g. by doing it when (if) we sanitize segments
            # Append to segment list for file
            # end for
        # end for

        # Sanitize segments w.r.t all files
        # Trim segments
        ...


class SaccadeAmplitudeNormalizer(FileProcessor):
    """
    Normalizes files individually based on the approximate median amplitude of all positive saccades.
    """

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

        for i in range(len(frames)):
            target = frames[i]["target"]
            target_diff = target.diff().fillna(0)
            anchors = [*target[target_diff != 0].index.tolist()]
            if not len(anchors):
                del frames[i]
                skipped += 1
                continue

            saccade_diffs = []
            for j in range(0, len(anchors) - 1, 2):
                # Use only positive values, as negative and positive saccades have different magnitude in eye position.
                if target_diff[anchors[j]] < 0:
                    continue
                saccade = frames[i]["position"][anchors[j] - 1: anchors[j + 1]]
                peak_value = np.median(saccade[-self.n:])
                idle_value = np.median(saccade[:self.n])
                saccade_diffs.append(abs(peak_value - idle_value))
            if not len(saccade_diffs):
                raise Exception("no positive saccades found")
            # scale the reference value to get a nice range
            frames[i][["position", "drift", "target"]] /= (np.median(saccade_diffs) * self.scaling_factor)

        print(f"skipped {skipped} files (no target movement)")
        return frames


class FileFilter(FileProcessor):
    """
    Filters files based on z scores on a number of heuristics.
    """
    SERIES_HEADERS = ['position', 'drift']

    def __init__(self, *, position_threshold, drift_thresholds):
        self.position_thresholds = position_threshold.values()
        self.drift_thresholds = drift_thresholds.values()

    def __call__(self, frames: List[DataFrame], **kwargs) -> List[DataFrame]:
        """
        Filter the frames by max value, signal-to-noise ratio and mean velocity. The filter is applied with respect to
        the position and the drift independently.

        :param frames: A list holding the dataframes to be filtered
        """
        for header, thresholds in zip(FileFilter.SERIES_HEADERS, [self.position_thresholds, self.drift_thresholds]):
            series = [df[header] for df in frames]
            heuristics = {
                # OBS I now added the abs before max which wasn't there before
                'max value': [s.abs().max() for s in series],
                'snr': [s.abs().max() / (s.var() + finfo(float).eps) for s in series],
                # TODO (low prio) we could compute the velocity before this point and use that. That way we would
                #  avoid computing diff here AND later when adding the velocity channel.
                'mean velocity': [s.diff().fillna(0).abs().mean(axis=0) for s in series]
            }

            for (h_name, h_values), threshold in zip(heuristics.items(), thresholds):
                z_filter = ZScoreFilter(threshold)
                n_before = len(frames)
                # Filter all remaining frames based on the computed heuristic
                frames = [f for f, support in zip(frames, h_values) if not z_filter(support)]
                print(f'skipped {len(frames) - n_before} frames ({header} {h_name} outlier)')

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
        for i in range(0, len(anchors) - 1, 2):
            s = frame.iloc[anchors[i] + th:anchors[i + 1], :].copy()
            # Add the initial timestamp of the segment
            start_time = s["Time (ms)"].iloc[0]
            s["Time (ms)"] -= start_time
            s.reset_index(drop=True, inplace=True)
            # Place the segment in the returning list
            segments.append(s)

        return segments[(1 if self.exclude_first else 0):]
