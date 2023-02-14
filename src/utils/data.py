from scipy.stats import zscore
import numpy as np
import torch


class PadCollate:
    """
    a variant of collate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def __call__(self, batch):
        """
        args:
            batch - list of (tensor, label)

        return:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # pad according to max_len
        batch = list(map(lambda xy: (self.__class__.pad_tensor(xy[0], pad=max_len, dim=self.dim), xy[1]), batch))
        # stack all
        xs = torch.stack([x[0] for x in batch], dim=0)
        ys = torch.stack([x[1] for x in batch], dim=0)
        return xs, ys

    @staticmethod
    def pad_tensor(vec, pad, dim):
        """
        Parameters
        ----------
        vec: tensor to pad
        pad: the size to pad to
        dim: dimension to pad

        Returns
        -------
        a new tensor padded to 'pad' in dimension 'dim'
        """
        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


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
        df[col] = df[col].mask(z_score_outlier(df[col])).interpolate()
    return df
