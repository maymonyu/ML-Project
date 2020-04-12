
import numpy as np

import torch
import torch.utils.data
from tqdm import tqdm_notebook as tqdm

from OurModel.Models.Batch import Batch


def log_norm_pdf(x, mu, logvar):
    return -0.5*(logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


def log_norm_std_pdf(x):
    return -0.5*(np.log(2 * np.pi) + x.pow(2))


def swish_(x):
    return x.mul_(torch.sigmoid(x))


def swish(x):
    return x.mul(torch.sigmoid(x))


def kl(q_distr, p_distr, weights, eps=1e-7):
    mu_q, logvar_q = q_distr
    mu_p, logvar_p = p_distr
    return 0.5 * (((logvar_q.exp() + (mu_q - mu_p).pow(2)) / (logvar_p.exp() + eps) \
                    + logvar_p - logvar_q - 1
                   ).sum(dim=-1) * weights).mean()


def simple_kl(mu_q, logvar_q, logvar_p_scale, norm):
    return (-0.5 * ( (1 + logvar_q #- torch.log(torch.ones(1)*logvar_p_scale) \
                      - mu_q.pow(2)/logvar_p_scale - logvar_q.exp()/logvar_p_scale
                     )
                   ).sum(dim=-1) * norm
           ).mean()


def log_norm_pdf(x, mu, logvar):
    return -0.5*(logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


def log_norm_std_pdf(x):
    return -0.5*(np.log(2 * np.pi) + x.pow(2))


def batch_data_sampler(batch_size, device, axis, input_data, output_data=None, shuffle=False, samples_perc_per_epoch=1):
    assert axis in ['users', 'items']
    assert 0 < samples_perc_per_epoch <= 1

    if axis == 'items':
        input_data = input_data.T
        if output_data is not None:
            output_data = output_data.T

    total_samples = input_data.shape[0]
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

        yield Batch(device, idx, input_data, output_data)
