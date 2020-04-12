import numpy as np

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

from tqdm import tqdm_notebook as tqdm

from OurModel.Models.GaussianMixturePriorWithAprPost import GaussianMixturePriorWithAprPost
from OurModel.Models.DeterministicDecoder import DeterministicDecoder
from OurModel.Models.MLFunctions import swish, log_norm_pdf, batch_data_sampler
from OurModel.Models.Batch import Batch


class VAE(nn.Module):
    def __init__(self, model_params, input_dim, axis, device):
        super(VAE, self).__init__()
        hidden_dim, latent_dim, lr, encoder_opt_type, decoder_opt_type, embedding_opt_type = model_params

        self.fc1 = nn.Linear(input_dim[1], hidden_dim)
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

        self.prior = GaussianMixturePriorWithAprPost(latent_dim, input_dim[0])
        self.decoder = DeterministicDecoder(latent_dim, input_dim[1])

        self.axis = axis

        self.device = device

        self.lr = lr

        decoder_params = set(self.decoder.parameters())
        embedding_params = set(self.prior.user_mu.parameters()) | set(self.prior.user_logvar.parameters())
        encoder_params = set(self.parameters()) - decoder_params - embedding_params

        self.optimizer_encoder = encoder_opt_type(encoder_params, lr=self.lr)
        self.optimizer_decoder = decoder_opt_type(decoder_params, lr=self.lr)
        self.optimizer_embedding = embedding_opt_type(embedding_params, lr=self.lr)

        self.opts = [self.optimizer_encoder, self.optimizer_decoder, self.optimizer_embedding]

        print('encoder\n', [x.shape for x in encoder_params])
        print('embedding\n', [x.shape for x in embedding_params])
        print('decoder\n', [x.shape for x in decoder_params])

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

        for batch in batch_data_sampler(batch_size=500, device=self.device, input_data=train_data, axis=self.axis):

            user_ratings = batch.get_ratings_to_device()
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
