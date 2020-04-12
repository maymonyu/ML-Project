import numpy as np
# from scipy import sparse
# %matplotlib inline

import torch
import torch.utils.data
from torch import nn

from OurModel.Models.MLFunctions import log_norm_pdf


class GaussianMixturePriorWithAprPost(nn.Module):
    def __init__(self, latent_dim, input_count):
        super(GaussianMixturePriorWithAprPost, self).__init__()

        self.gaussians_number = 1

        self.mu_prior = nn.Parameter(torch.Tensor(latent_dim, self.gaussians_number))
        self.mu_prior.data.fill_(0)

        self.logvar_prior = nn.Parameter(torch.Tensor(latent_dim, self.gaussians_number))
        self.logvar_prior.data.fill_(0)

        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(latent_dim, self.gaussians_number))
        self.logvar_uniform_prior.data.fill_(10)

        self.user_mu = nn.Embedding(input_count, latent_dim)
        self.user_logvar = nn.Embedding(input_count, latent_dim)

    def forward(self, z, idx):
        density_per_gaussian1 = log_norm_pdf(x=z[:, :, None],
                                             mu=self.mu_prior[None, :, :].detach(),
                                             logvar=self.logvar_prior[None, :, :].detach()
                                             ).add(np.log( 1 /5 - 1/ 20))

        density_per_gaussian2 = log_norm_pdf(x=z[:, :, None],
                                             mu=self.user_mu(idx)[:, :, None].detach(),
                                             logvar=self.user_logvar(idx)[:, :, None].detach()
                                             ).add(np.log(4 / 5 - 1 / 20))

        density_per_gaussian3 = log_norm_pdf(x=z[:, :, None],
                                             mu=self.mu_prior[None, :, :].detach(),
                                             logvar=self.logvar_uniform_prior[None, :, :].detach()
                                             ).add(np.log(1 / 10))

        density_per_gaussian = torch.cat([density_per_gaussian1,
                                          density_per_gaussian2,
                                          density_per_gaussian3], dim=-1)

        return torch.logsumexp(density_per_gaussian, dim=-1)
