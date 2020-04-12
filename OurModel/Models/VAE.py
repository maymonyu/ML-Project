import numpy as np

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

from tqdm import tqdm_notebook as tqdm

from OurModel.Models.GaussianMixturePriorWithAprPost import GaussianMixturePriorWithAprPost
from OurModel.Models.DeterministicDecoder import DeterministicDecoder
from OurModel.Models.MLFunctions import swish, log_norm_pdf
from OurModel.Models.Batch import Batch


class VAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, matrix_dim, axis, device):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(matrix_dim[1], hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

        self.prior = GaussianMixturePriorWithAprPost(latent_dim, matrix_dim[0])
        self.decoder = DeterministicDecoder(latent_dim, matrix_dim[1])

        self.axis = axis

        self.device = device

    def encode(self, x, dropout_rate):
        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]

        x = F.dropout(x, p=dropout_rate, training=self.training)

        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc21(h5), self.fc22(h5)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 *logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, user_ratings, user_idx, beta=1, dropout_rate=0.5, calculate_loss=True, mode=None):

        if mode == 'pr':
            mu, logvar = self.encode(user_ratings, dropout_rate=dropout_rate)
        elif mode == 'mf':
            mu, logvar = self.encode(user_ratings, dropout_rate=0)

        z = self.reparameterize(mu, logvar)
        x_pred, decoder_loss = self.decode(z)

        NLL = -(F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()

        if calculate_loss:
            if mode == 'pr':
                norm = user_ratings.sum(dim=-1)
                KLD = -(self.prior(z, user_idx) - log_norm_pdf(z, mu, logvar)).sum(dim=-1).mul(norm).mean()
                loss = NLL + beta * KLD + decoder_loss

            elif mode == 'mf':
                KLD = NLL * 0
                loss = NLL + decoder_loss

            return (NLL, KLD), loss

        else:
            return x_pred

    def set_embeddings(self, train_data, momentum=0, weight=None):
        istraining = self.training
        self.eval()

        for batch in generate(batch_size=500, device=self.device, data_1=train_data, axis=self.axis):

            user_ratings = batch.get_ratings_to_dev()
            users_idx = batch.get_idx()

            new_user_mu, new_user_logvar = self.encode(user_ratings, 0)

            old_user_mu = self.prior.user_mu.weight.data[users_idx, :].detach()
            old_user_logvar = self.prior.user_logvar.weight.data[users_idx, :].detach()

            if weight:
                old_user_var = torch.exp(old_user_logvar)
                new_user_var = torch.exp(new_user_logvar)

                post_user_var = 1 / (1 / old_user_var + weight / new_user_var)
                post_user_mu = (old_user_mu / old_user_var + weight * new_user_mu / new_user_var) * post_user_var

                self.prior.user_mu.weight.data[users_idx, :] = post_user_mu
                self.prior.user_logvar.weight.data[users_idx, :] = torch.log(post_user_var + new_user_var)
            else:
                self.prior.user_mu.weight.data[users_idx, :] = momentum * old_user_mu + ( 1 -momentum) * new_user_mu
                self.prior.user_logvar.weight.data[users_idx, :] = momentum * old_user_logvar + \
                            (1 - momentum) * new_user_logvar

        if istraining:
            self.train()
        else:
            self.eval()


def generate(batch_size, device, axis, data_1, data_2=None, shuffle=False, samples_perc_per_epoch=1):
    assert axis in ['users', 'items']
    assert 0 < samples_perc_per_epoch <= 1

    if axis == 'items':
        data_1 = data_1.T
        if data_2 is not None:
            data_2 = data_2.T

    total_samples = data_1.shape[0]
    samples_per_epoch = int(total_samples * samples_perc_per_epoch)

    if shuffle:
        idxlist = np.arange(total_samples)
        np.random.shuffle(idxlist)
        idxlist = idxlist[:samples_per_epoch]
    else:
        idxlist = np.arange(samples_per_epoch)

    for st_idx in tqdm(range(0, samples_per_epoch, batch_size)):
        end_idx = min(st_idx + batch_size, samples_per_epoch)
        idx = idxlist[st_idx:end_idx]

        yield Batch(device, idx, data_1, data_2)
