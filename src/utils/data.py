import numpy as np
from scipy.stats import zscore


class ZScoreFilter:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, data):
        return z_score_outlier(data, self.threshold)


def z_score_outlier(data, std=3.0):
    z_scores = np.abs(zscore(data, axis=0))
    return z_scores > std
