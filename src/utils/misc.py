import random

import numpy as np
import torch.random


def set_random_state(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
