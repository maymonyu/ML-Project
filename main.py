import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import time
import json
import pickle
import random
import functools
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime as dt
import matplotlib.pyplot as plt

import warnings

from preprocess import preprocess
from utils import clear_log_file, file_write, load_obj, load_obj_json, plot_len_vs_ndcg, plt,\
    pretty_print, save_obj, save_obj_json
from dataLoader import load_data
from Model.VAELoss import VAELoss
from Model.Model import Model

warnings.filterwarnings('ignore')

# # Hyper Parameters

hyper_params = {
    'data_base': 'data/ml-1m/',
    'project_name': 'svae_ml1m',
    'model_file_name': '',
    'log_file': '',
    'history_split_test': [0.8, 0.2],

    'learning_rate': 0.01,  # learning rate is required only if optimizer is adagrad
    'optimizer': 'adam',
    'weight_decay': float(5e-3),

    'epochs': 25,
    'batch_size': 1,  # Needs to be 1, because we don't pack multiple sequences in the same batch

    'item_embed_size': 256,
    'rnn_size': 200,
    'hidden_size': 150,
    'latent_size': 64,
    'loss_type': 'next_k',  # [predict_next, same, prefix, postfix, exp_decay, next_k]
    'next_k': 4,

    'number_users_to_keep': 1000000000,
    'batch_log_interval': 1000,
    'train_cp_users': 200,
    'exploding_clip': 0.25,
}

file_name = '_optimizer_' + str(hyper_params['optimizer'])
if hyper_params['optimizer'] == 'adagrad':
    file_name += '_lr_' + str(hyper_params['learning_rate'])
file_name += '_weight_decay_' + str(hyper_params['weight_decay'])
file_name += '_loss_type_' + str(hyper_params['loss_type'])
file_name += '_item_embed_size_' + str(hyper_params['item_embed_size'])
file_name += '_rnn_size_' + str(hyper_params['rnn_size'])
file_name += '_latent_size_' + str(hyper_params['latent_size'])

log_file_root = "saved_logs/"  # Don't remove the '/' at the end please :)
model_file_root = "saved_models/"  # Don't remove the '/' at the end please :)

if not os.path.isdir(log_file_root): os.mkdir(log_file_root)
if not os.path.isdir(model_file_root): os.mkdir(model_file_root)
hyper_params['log_file'] = log_file_root + hyper_params['project_name'] + '_log' + file_name + '.txt'
hyper_params['model_file_name'] = model_file_root + hyper_params['project_name'] + '_model' + file_name + '.pt'

preprocess(hyper_params)

# --------------------------------------------------------------------------------------------------

LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor

is_cuda_available = torch.cuda.is_available()

if is_cuda_available:
    print("Using CUDA...\n")
    LongTensor = torch.cuda.LongTensor
    FloatTensor = torch.cuda.FloatTensor

# # Evaluation Code


def evaluate(model, criterion, reader, hyper_params, is_train_set):
    model.eval()

    metrics = {}
    metrics['loss'] = 0.0
    Ks = [10, 100]
    for k in Ks:
        metrics['NDCG@' + str(k)] = 0.0
        metrics['Rec@' + str(k)] = 0.0
        metrics['Prec@' + str(k)] = 0.0

    batch = 0
    total_users = 0.0

    # For plotting the results (seq length vs. NDCG@100)
    len_to_ndcg_at_100_map = {}

    for x, y_s, test_movies, test_movies_r in reader.iter_eval():
        batch += 1
        if is_train_set == True and batch > hyper_params['train_cp_users']: break

        decoder_output, z_mean, z_log_sigma = model(x)

        metrics['loss'] += criterion(decoder_output, z_mean, z_log_sigma, y_s, 0.2).data

        # Making the logits of previous items in the sequence to be "- infinity"
        decoder_output = decoder_output.data
        x_scattered = torch.zeros(decoder_output.shape[0], decoder_output.shape[2])
        if is_cuda_available: x_scattered = x_scattered.cuda()
        x_scattered[0, :].scatter_(0, x[0].data, 1.0)
        last_predictions = decoder_output[:, -1, :] - (torch.abs(decoder_output[:, -1, :] * x_scattered) * 100000000)

        for batch_num in range(
                last_predictions.shape[0]):  # batch_num is ideally only 0, since batch_size is enforced to be always 1
            predicted_scores = last_predictions[batch_num]
            actual_movies_watched = test_movies[batch_num]
            actual_movies_ratings = test_movies_r[batch_num]

            # Calculate NDCG
            _, argsorted = torch.sort(-1.0 * predicted_scores)
            for k in Ks:
                best, now_at, dcg, hits = 0.0, 0.0, 0.0, 0.0

                rec_list = list(argsorted[:k].cpu().numpy())
                for m in range(len(actual_movies_watched)):
                    movie = actual_movies_watched[m]
                    now_at += 1.0
                    if now_at <= k: best += 1.0 / float(np.log2(now_at + 1))

                    if movie not in rec_list: continue
                    hits += 1.0
                    dcg += 1.0 / float(np.log2(float(rec_list.index(movie) + 2)))

                metrics['NDCG@' + str(k)] += float(dcg) / float(best)
                metrics['Rec@' + str(k)] += float(hits) / float(len(actual_movies_watched))
                metrics['Prec@' + str(k)] += float(hits) / float(k)

                # Only for plotting the graph (seq length vs. NDCG@100)
                if k == 100:
                    seq_len = int(len(actual_movies_watched)) + int(x[batch_num].shape[0]) + 1
                    if seq_len not in len_to_ndcg_at_100_map: len_to_ndcg_at_100_map[seq_len] = []
                    len_to_ndcg_at_100_map[seq_len].append(float(dcg) / float(best))

            total_users += 1.0

    metrics['loss'] = float(metrics['loss']) / float(batch)
    metrics['loss'] = round(metrics['loss'], 4)

    for k in Ks:
        metrics['NDCG@' + str(k)] = round((100.0 * metrics['NDCG@' + str(k)]) / float(total_users), 4)
        metrics['Rec@' + str(k)] = round((100.0 * metrics['Rec@' + str(k)]) / float(total_users), 4)
        metrics['Prec@' + str(k)] = round((100.0 * metrics['Prec@' + str(k)]) / float(total_users), 4)

    return metrics, len_to_ndcg_at_100_map

# # Training loop


def train(reader):
    model.train()
    total_loss = 0
    start_time = time.time()
    batch = 0
    batch_limit = int(train_reader.num_b)
    total_anneal_steps = 200000
    anneal = 0.0
    update_count = 0.0
    anneal_cap = 0.2

    for x, y_s in reader.iter():
        batch += 1

        # Empty the gradients
        model.zero_grad()
        optimizer.zero_grad()

        # Forward pass
        decoder_output, z_mean, z_log_sigma = model(x)

        # Backward pass
        loss = criterion(decoder_output, z_mean, z_log_sigma, y_s, anneal)
        loss.backward()
        optimizer.step()

        total_loss += loss.data

        # Anneal logic
        if total_anneal_steps > 0:
            anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
        else:
            anneal = anneal_cap
        update_count += 1.0

        # Logging mechanism
        if (batch % hyper_params['batch_log_interval'] == 0 and batch > 0) or batch == batch_limit:
            div = hyper_params['batch_log_interval']
            if batch == batch_limit: div = (batch_limit % hyper_params['batch_log_interval']) - 1
            if div <= 0: div = 1

            cur_loss = (total_loss / div)
            elapsed = time.time() - start_time

            ss = '| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.4f}'.format(
                epoch, batch, batch_limit, (elapsed * 1000) / div, cur_loss
            )

            file_write(hyper_params['log_file'], ss)

            total_loss = 0
            start_time = time.time()

def get_optimizer(hyper_params):
    if hyper_params['optimizer'] == 'adagrad':
        optimizer = torch.optim.Adagrad(
            model.parameters(), weight_decay=hyper_params['weight_decay'], lr=hyper_params['learning_rate']
        )
    elif hyper_params['optimizer'] == 'adadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(), weight_decay=hyper_params['weight_decay']
        )
    elif hyper_params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), weight_decay=hyper_params['weight_decay']
        )
    elif hyper_params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(), weight_decay=hyper_params['weight_decay']
        )
    return optimizer


# Train It..
train_reader, val_reader, test_reader, total_items = load_data(hyper_params, is_cuda_available, LongTensor)
hyper_params['total_items'] = total_items
hyper_params['testing_batch_limit'] = test_reader.num_b

file_write(hyper_params['log_file'], "\n\nSimulation run on: " + str(dt.datetime.now()) + "\n\n")
file_write(hyper_params['log_file'], "Data reading complete!")
file_write(hyper_params['log_file'], "Number of train batches: {:4d}".format(train_reader.num_b))
file_write(hyper_params['log_file'], "Number of validation batches: {:4d}".format(val_reader.num_b))
file_write(hyper_params['log_file'], "Number of test batches: {:4d}".format(test_reader.num_b))
file_write(hyper_params['log_file'], "Total Items: " + str(total_items) + "\n")

model = Model(hyper_params, is_cuda_available)
if is_cuda_available: model.cuda()

criterion = VAELoss(hyper_params)
optimizer = get_optimizer(hyper_params)

file_write(hyper_params['log_file'], str(model))
file_write(hyper_params['log_file'], "\nModel Built!\nStarting Training...\n")

best_val_ndcg = None

try:
    for epoch in range(1, hyper_params['epochs'] + 1):
        epoch_start_time = time.time()

        train(train_reader)

        # Calulating the metrics on the train set
        metrics, _ = evaluate(model, criterion, train_reader, hyper_params, True)
        string = ""
        for m in metrics: string += " | " + m + ' = ' + str(metrics[m])
        string += ' (TRAIN)'

        # Calulating the metrics on the validation set
        metrics, _ = evaluate(model, criterion, val_reader, hyper_params, False)
        string2 = ""
        for m in metrics: string2 += " | " + m + ' = ' + str(metrics[m])
        string2 += ' (VAL)'

        ss = '-' * 89
        ss += '\n| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time))
        ss += string
        ss += '\n'
        ss += '-' * 89
        ss += '\n| end of epoch {:3d} | time: {:5.2f}s'.format(epoch, (time.time() - epoch_start_time))
        ss += string2
        ss += '\n'
        ss += '-' * 89
        file_write(hyper_params['log_file'], ss)

        if not best_val_ndcg or metrics['NDCG@100'] >= best_val_ndcg:
            with open(hyper_params['model_file_name'], 'wb') as f: torch.save(model, f)
            best_val_ndcg = metrics['NDCG@100']

except KeyboardInterrupt:
    print('Exiting from training early')

# Plot Traning graph
f = open(model.hyper_params['log_file'])
lines = f.readlines()
lines.reverse()

train = []
test = []

for line in lines:
    if line[:10] == 'Simulation' and len(train) > 1:
        break
    elif line[:10] == 'Simulation' and len(train) <= 1:
        train, test = [], []

    if line[2:5] == 'end' and line[-5:-2] == 'VAL':
        test.append(line.strip().split("|"))
    elif line[2:5] == 'end' and line[-7:-2] == 'TRAIN':
        train.append(line.strip().split("|"))

train.reverse()
test.reverse()

train_ndcg = []
test_ndcg = []
test_loss, train_loss = [], []

for i in train:
    for metric in i:
        if metric.split("=")[0] == " NDCG@100 ":
            train_ndcg.append(float(metric.split('=')[1].split(' ')[1]))
        if metric.split("=")[0] == " loss ":
            train_loss.append(float(metric.split("=")[1].split(' ')[1]))

total, avg_runtime = 0.0, 0.0
for i in test:
    avg_runtime += float(i[2].split(" ")[2][:-1])
    total += 1.0

    for metric in i:
        if metric.split("=")[0] == " NDCG@100 ":
            test_ndcg.append(float(metric.split('=')[1].split(' ')[1]))
        if metric.split("=")[0] == " loss ":
            test_loss.append(float(metric.split("=")[1].split(' ')[1]))

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.set_title(hyper_params["project_name"], fontweight="bold", size=20)
ax1.plot(test_ndcg, 'b-')
ax1.set_xlabel('Epochs', fontsize=20.0)
ax1.set_ylabel('NDCG@100', color='b', fontsize=20.0)
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(test_loss, 'r--')
ax2.set_ylabel('Loss', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
if not os.path.isdir("saved_plots/"): os.mkdir("saved_plots/")
fig.savefig("saved_plots/learning_curve_" + hyper_params["project_name"] + ".pdf")
plt.show()

# Checking metrics for the test set on best saved model
with open(hyper_params['model_file_name'], 'rb') as f: model = torch.load(f)
metrics, len_to_ndcg_at_100_map = evaluate(model, criterion, test_reader, hyper_params, False)

# Plot sequence length vs NDCG@100 graph
plot_len_vs_ndcg(len_to_ndcg_at_100_map, hyper_params)

string = ""
for m in metrics: string += " | " + m + ' = ' + str(metrics[m])

ss = '=' * 89
ss += '\n| End of training'
ss += string + " (TEST)"
ss += '\n'
ss += '=' * 89
file_write(hyper_params['log_file'], ss)
print("average runtime per epoch =", round(avg_runtime / float(total), 4), "s")
