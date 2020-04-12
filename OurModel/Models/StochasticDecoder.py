import numpy as np

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

from tqdm import tqdm_notebook as tqdm


class StochasticDecoder(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(StochasticDecoder, self).__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.logvar = nn.Parameter(torch.Tensor(out_features, in_features))
        self.logvar.data.fill_(-2)

    def forward(self, input):

        if self.training:
            std = torch.exp(self.logvar)
            a = F.linear(input, self.weight, self.bias)
            eps = torch.randn_like(a)
            b = eps.mul_(torch.sqrt_(F.linear(input * input, std)))
            output = a + b

            kl = (-0.5 * (1 + self.logvar - self.weight.pow(2) - self.logvar.exp())).sum(dim=-1).mean()  # / (10)
            return output, kl
        else:
            output = F.linear(input, self.weight, self.bias)
            return output, 0
