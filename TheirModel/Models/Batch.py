import inline as inline
import matplotlib
import numpy as np
# from scipy import sparse
import matplotlib.pyplot as plt
# %matplotlib inline

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

from tqdm import tqdm_notebook as tqdm
import pickle
import random
from IPython.display import clear_output


class Batch:
    def __init__(self, device, idx, data_1, data_2=None):
        self._device = device
        self._idx = idx
        self._data_1 = data_1
        self._data_2 = data_2

    def get_idx(self):
        return self._idx

    def get_idx_to_dev(self):
        return torch.LongTensor(self.get_idx()).to(self._device)

    def get_ratings(self, is_test=False):
        data = self._data_2 if is_test else self._data_1
        return data[self._idx]

    def get_ratings_to_dev(self, is_test=False):
        return torch.Tensor(
            self.get_ratings(is_test).toarray()
        ).to(self._device)
