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

import utils


seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda")

data = (x.astype('float32') for x in utils.get_data(global_indexing=False, dataset='pro_sg'))
train_data, valid_1_data, valid_2_data, test_1_data, test_2_data = data
n_users, n_items = train_data.shape

print(n_items, n_users)
