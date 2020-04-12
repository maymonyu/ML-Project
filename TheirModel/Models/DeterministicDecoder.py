import numpy as np

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

from tqdm import tqdm_notebook as tqdm


class DeterministicDecoder(nn.Linear):
    def __init__(self, *args):
        super(DeterministicDecoder, self).__init__(*args)

    def forward(self, *args):
        output = super(DeterministicDecoder, self).forward(*args)
        return output, 0
