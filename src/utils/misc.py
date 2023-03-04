import random

import numpy as np
import torch.random


def set_random_state(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def torch_unique_index(x: torch.Tensor, dim=-1):
    """
    Source: https://github.com/pytorch/pytorch/issues/36748#issuecomment-1072093200
    """
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)
