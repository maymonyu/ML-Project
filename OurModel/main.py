import numpy as np
# from scipy import sparse
import matplotlib.pyplot as plt
# %matplotlib inline

import torch.utils.data
from torch import optim

import random
from IPython.display import clear_output

from TheirModel import utils
from TheirModel.Models.VAE import VAE
from TheirModel.Models.MLFunctions import generate

seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda")

data = (x.astype('float32') for x in utils.get_data(global_indexing=False, dataset='pro_sg'))
train_data, valid_1_data, valid_2_data, test_1_data, test_2_data = data
n_users, n_items = train_data.shape

print(n_items, n_users)


# def get_notebook_name():
#     kernel_id = re.search('kernel-(.*).json', ipykernel.connect.get_connection_file()).group(1)
#     servers = list_running_servers()
#     for ss in servers:
#         response = requests.get(urljoin(ss['url'], 'api/sessions'), params={'token': ss.get('token', '')})
#         for nn in json.loads(response.text):
#             if nn['kernel']['id'] == kernel_id:
#                 relative_path = nn['notebook']['path']
#                 return relative_path.split('/')[-1].split('.')[0]
#
#
# ser_model_fn = 'model_' + get_notebook_name().replace(' ', '_') + '.pt'
# print(ser_model_fn)


def validate(model, data_1, data_2, axis, mode, samples_perc_per_epoch=1):
    model.eval()
    batch_size = 500
    ndcg_dist = []

    for batch in generate(batch_size=batch_size,
                          device=device,
                          axis=axis,
                          data_1=data_1,
                          data_2=data_2,
                          samples_perc_per_epoch=samples_perc_per_epoch
                          ):

        ratings = batch.get_ratings_to_dev()
        idx = batch.get_idx_to_dev()
        ratings_test = batch.get_ratings(is_test=True)

        pred_val = model(ratings, idx, calculate_loss=False, mode=mode).cpu().detach().numpy()

        if not (data_1 is data_2):
            pred_val[batch.get_ratings().nonzero()] = -np.inf
        ndcg_dist.append(utils.NDCG_binary_at_k_batch(pred_val, ratings_test))

    ndcg_dist = np.concatenate(ndcg_dist)
    return ndcg_dist[~np.isnan(ndcg_dist)].mean()


def run(model, opts, train_data, batch_size, n_epochs, axis, beta, mode):
    global best_ndcg
    global ndcgs_tr_pr, ndcgs_tr_mf, ndcgs_va_pr, ndcgs_va_mf

    for epoch in range(n_epochs):
        model.train()
        NLL_loss = 0
        KLD_loss = 0

        for batch in generate(batch_size=batch_size, device=device, axis=axis, data_1=train_data, shuffle=True):
            ratings = batch.get_ratings_to_dev()
            idx = batch.get_idx_to_dev()

            for optimizer in opts:
                optimizer.zero_grad()

            (NLL, KLD), loss = model(ratings, idx, beta=beta, mode=mode)
            loss.backward()

            for optimizer in opts:
                optimizer.step()

            NLL_loss += NLL.item()
            KLD_loss += KLD.item()

        print('NLL_loss', NLL_loss, 'KLD_loss', KLD_loss)


# ndcg = score (or loss or something like that)
best_ndcg = -np.inf
ndcgs_tr_pr, ndcgs_tr_mf, ndcgs_va_pr, ndcgs_va_mf = [], [], [], []
var_param_distance = []

hidden_dim = 600
latent_dim = 200

model_i = VAE(hidden_dim, latent_dim, (n_users, n_items), 'users', device).to(device)
model_i.set_embeddings(train_data)
print(model_i)


def get_opts(model, lr=5e-4):
    decoder_params = set(model.decoder.parameters())
    embedding_params = set(model_i.prior.user_mu.parameters()) | set(model_i.prior.user_logvar.parameters())
    encoder_params = set(model.parameters()) - decoder_params - embedding_params

    optimizer_encoder = optim.Adam(encoder_params, lr=lr)
    optimizer_decoder = optim.Adam(decoder_params, lr=lr)
    optimizer_embedding = optim.Adam(embedding_params, lr=lr)

    print('encoder\n', [x.shape for x in encoder_params])
    print('embedding\n', [x.shape for x in embedding_params])
    print('decoder\n', [x.shape for x in decoder_params])

    return optimizer_encoder, optimizer_decoder, optimizer_embedding


optimizer_encoder_i, optimizer_decoder_i, _ = get_opts(model_i)

best_model = model_i
for epoch in range(50):
    run(model_i, [optimizer_encoder_i],
        train_data, batch_size=500, n_epochs=3, axis='users', mode='pr', beta=0.005)
    model_i.set_embeddings(train_data)
    run(model_i, [optimizer_decoder_i],
        train_data, batch_size=500, n_epochs=1, axis='users', mode='mf', beta=None)

    model = model_i
    axis = 'users'
    ndcg_ = validate(model, train_data, train_data, axis, 'mf', 0.01)
    ndcgs_tr_mf.append(ndcg_)
    ndcg_ = validate(model, train_data, train_data, axis, 'pr', 0.01)
    ndcgs_tr_pr.append(ndcg_)
    ndcg_ = validate(model, valid_1_data, valid_2_data, axis, 'pr', 1)
    ndcgs_va_pr.append(ndcg_)

    clear_output(True)

    i_min = np.array(ndcgs_va_pr).argsort()[-len(ndcgs_va_pr) // 2:].min()

    print('ndcg', ndcgs_va_pr[-1], ': : :', best_ndcg)
    fig, ax1 = plt.subplots()
    fig.set_size_inches(15, 5)

    ax1.plot(range(i_min, len(ndcgs_va_pr)), ndcgs_va_pr[i_min:], '+-', label='pr valid')
    ax1.legend(loc='lower right')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(range(i_min, len(ndcgs_va_pr)), ndcgs_tr_pr[i_min:], '+:', label='pr train')
    ax2.plot(range(i_min, len(ndcgs_va_pr)), ndcgs_tr_mf[i_min:], 'x:', label='mf train')
    ax2.legend(loc='lower left')

    fig.tight_layout()
    plt.ylabel("Validation NDCG@100")
    plt.xlabel("Epochs")
    plt.show()

    if ndcg_ > best_ndcg:
        best_ndcg = ndcg_
        best_model = model
        # torch.save(model.state_dict(), ser_model_fn)

    if ndcg_ < best_ndcg / 2 and epoch > 10:
        break

# model_i.load_state_dict(torch.load(ser_model_fn))
model_i = best_model

batch_size_test = 2000
model_i.eval()
n100_list, r20_list, r50_list = [], [], []

for batch in generate(batch_size=batch_size_test, device=device, axis='users', data_1=test_1_data, data_2=test_2_data,
                      samples_perc_per_epoch=1):
    user_ratings = batch.get_ratings_to_dev()
    users_idx = batch.get_idx_to_dev()
    user_ratings_test = batch.get_ratings(is_test=True)

    pred_val = model_i(user_ratings, users_idx, calculate_loss=False, mode='mf').cpu().detach().numpy()
    # exclude examples from training and validation (if any)
    pred_val[batch.get_ratings().nonzero()] = -np.inf
    n100_list.append(utils.NDCG_binary_at_k_batch(pred_val, user_ratings_test, k=100))
    r20_list.append(utils.Recall_at_k_batch(pred_val, user_ratings_test, k=20))
    r50_list.append(utils.Recall_at_k_batch(pred_val, user_ratings_test, k=50))

n100_list = np.concatenate(n100_list)
r20_list = np.concatenate(r20_list)
r50_list = np.concatenate(r50_list)

print("Test NDCG@100=%.5f (%.5f)" % (
    np.mean(n100_list[~np.isnan(n100_list)]), np.std(n100_list) / np.sqrt(len(n100_list))))
print(
    "Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list[~np.isnan(r20_list)]), np.std(r20_list) / np.sqrt(len(r20_list))))
print(
    "Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list[~np.isnan(r50_list)]), np.std(r50_list) / np.sqrt(len(r50_list))))
