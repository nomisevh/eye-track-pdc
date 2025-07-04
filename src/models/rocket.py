import numpy as np
import torch
from torch import nn
from tqdm.autonotebook import tqdm

from utils.data import normalize as normalize_tensor


class ROCKET(nn.Module):
    """
    RandOm Convolutional KErnel Transform
    ROCKET is a GPU Pytorch implementation of the ROCKET functions generate_kernels
    and apply_kernels that can be used  with univariate and multivariate time series.
    Code Source: https://github.com/timeseriesAI/tsai/blob/main/tsai/models/ROCKET_Pytorch.py
    """

    def __init__(self, c_in, seq_len, n_kernels=10_000, kss=[7, 9, 11], device=None, verbose=False, normalize=False):
        """
        Input: is a 3d torch tensor of type torch.float32. When used with univariate TS,
        make sure you transform the 2d to 3d by adding unsqueeze(1).
        c_in: number of channels or features. For univariate c_in is 1.
        seq_len: sequence length
        """
        super().__init__()
        if device is None:
            device = 'cpu'
        kss = [ks for ks in kss if ks < seq_len]
        convs = nn.ModuleList()
        for i in range(n_kernels):
            ks = np.random.choice(kss)
            dilation = 2 ** np.random.uniform(0, np.log2((seq_len - 1) // (ks - 1)))
            padding = int((ks - 1) * dilation // 2) if np.random.randint(2) == 1 else 0
            weight = torch.randn(1, c_in, ks)
            weight -= weight.mean()
            bias = 2 * (torch.rand(1) - .5)
            layer = nn.Conv1d(c_in, 1, ks, padding=2 * padding, dilation=int(dilation), bias=True)
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            layer.bias = torch.nn.Parameter(bias, requires_grad=False)
            convs.append(layer)
        self.convs = convs
        self.n_kernels = n_kernels
        self.kss = kss
        # self.to(device=device)
        self.verbose = verbose
        self.normalize = normalize
        self.out_dim = 2 * n_kernels
        self.train = True
        self.mean = 0
        self.std = 0.01

    def forward(self, x):
        _output = []
        for i in tqdm(range(self.n_kernels), disable=not self.verbose, leave=False, desc='kernel/kernels'):
            out = self.convs[i](x.float())
            _max = out.max(dim=-1)[0]
            _ppv = torch.gt(out, 0).sum(dim=-1).float() / out.shape[-1]
            if not hasattr(self.convs[i], 'use_features') or self.convs[i].use_features[0]:
                _output.append(_max)
            if not hasattr(self.convs[i], 'use_features') or self.convs[i].use_features[1]:
                _output.append(_ppv)
        out = torch.cat(_output, dim=1)
        if self.normalize:
            if self.train:
                self.mean, self.std = out.mean(0, keepdim=True), out.std(0, unbiased=False, keepdim=True)
            return normalize_tensor(out, self.mean, self.std)
        return out


def dissected_forward(self, x):
    _output = []
    _output_ts = []
    for i in tqdm(range(self.n_kernels), disable=not self.verbose, leave=False, desc='kernel/kernels'):
        out = self.convs[i](x.float())
        _max = out.max(dim=-1)[0]
        _ppv = torch.gt(out, 0).sum(dim=-1).float() / out.shape[-1]
        _output.append(_max)
        _output.append(_ppv)
        _output_ts.append(out)
    out = torch.cat(_output, dim=1)
    # TODO for now trimming from the front to ensure all ts are the same length, but still end at the stimuli
    min_ts_length = min(t.shape[2] for t in _output_ts)
    _output_ts = [t[:, :, -min_ts_length:] for t in _output_ts]
    out_ts = torch.cat(_output_ts, dim=1)
    if self.normalize:
        return normalize_tensor(out), out_ts
    return out, out_ts
