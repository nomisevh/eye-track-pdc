import numpy as np
from numpy import median, logical_or
from scipy.signal import butter, filtfilt
from scipy.stats import zscore, median_absolute_deviation


class ZScoreMask:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, data):
        return med_mad_outlier(data, self.threshold)


def med_mad_outlier(data, deviations=3.0):
    med = median(data)
    mad = median_absolute_deviation(data)
    return logical_or(data > med + deviations * mad, data < med - deviations * mad)


def z_score_outlier(data, std=3.0):
    """
    :param data: The data for which z-scores are computed
    :param std: The threshold measured in number of std deviations away from the mean.
    :return: A mask over the data. True where the z-score is greater than the threshold.
    """
    z_scores = np.abs(zscore(data, axis=0))
    return z_scores > std


def interpolate_outliers(df, columns):
    for col in columns:
        df[col] = df[col].mask(z_score_outlier(df[col])).interpolate(limit_direction='both')
    return df


def binarize(dataset):
    dataset.y[dataset.y != 0] = 1


def normalize(x, m=None, s=None):
    if m is None:
        m = x.mean(0, keepdim=True)
    if s is None:
        s = x.std(0, unbiased=False, keepdim=True)
    x -= m
    x /= s
    return x


def butter_highpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False, output='ba')
    return b, a


def butter_highpass_filter(data, cutoff, fs, order):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y
