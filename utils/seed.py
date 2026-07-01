import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic cudnn trades a bit of speed for reproducibility - worth it given
    # how much staged-training comparison you'll likely be doing across configs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
