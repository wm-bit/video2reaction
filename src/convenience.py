from IPython import get_ipython

def activate_autoreload():
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
    print("In IPython")
    print("Set autoreload")


ipython = get_ipython()
if ipython is not None:
    print("In IPython")
    IN_IPYTHON = True
    activate_autoreload()
    # Code to automatically update the EasyTransformer code as its edited without restarting the kernel
    import tqdm.notebook as tqdm
else:
    print("Not in IPython")
    IN_IPYTHON = False
    import tqdm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Set seed to {seed}")