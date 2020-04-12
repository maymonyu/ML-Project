import numpy as np

import torch
import torch.utils.data
from torch import nn

from TheirModel.Models.MLFunctions import log_norm_pdf


class GaussianMixturePrior(nn.Module):
    def __init__(self, latent_dim, gaussians_number):
        super(GaussianMixturePrior, self).__init__()

        self.gaussians_number = gaussians_number

        self.mu_prior = nn.Parameter(torch.Tensor(latent_dim, gaussians_number))
        self.mu_prior.data.fill_(0)

        self.logvar_prior = nn.Parameter(torch.Tensor(latent_dim, gaussians_number))
        self.logvar_prior.data.fill_(0)

    def forward(self, z):
        density_per_gaussian = log_norm_pdf(x=z[:, :, None],
                                            mu=self.mu_prior[None, ...].detach(),
                                            logvar=self.logvar_prior[None, ...].detach()
                                            ).add(-np.log(self.gaussians_number))

        return torch.logsumexp(density_per_gaussian, dim=-1)
