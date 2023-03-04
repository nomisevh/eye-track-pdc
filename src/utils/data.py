import numpy as np
from scipy.stats import zscore


class ZScoreMask:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, data):
        return z_score_outlier(data, self.threshold)


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


def normalize(x):
    m = x.mean(0, keepdim=True)
    s = x.std(0, unbiased=False, keepdim=True)
    x -= m
    x /= s
    return x
