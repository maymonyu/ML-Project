import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
from torch import optim

import random
from IPython.display import clear_output

from OurModel import utils
from OurModel.Models.VAE import VAE
from OurModel.Models.MLFunctions import batch_data_sampler


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


def get_model_params():
    hidden_dim = 600
    latent_dim = 200
    lr = 5e-4
    encoder_opt_type = optim.Adam
    decoder_opt_type = optim.Adam
    embedding_opt_type = optim.Adam

    return [hidden_dim, latent_dim, lr,  encoder_opt_type, decoder_opt_type, embedding_opt_type]


def init_seed():
    seed = 1337
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate(model, input_data, output_data, axis, mode, samples_perc_per_epoch=1):
    model.eval()
    batch_size = 500
    ndcg_dist = []

    for batch in batch_data_sampler(batch_size=batch_size,
                                    device=device,
                                    axis=axis,
                                    input_data=input_data,
                                    output_data=output_data,
                                    samples_perc_per_epoch=samples_perc_per_epoch
                                    ):

        ratings = batch.get_ratings_to_device()
        idx = batch.get_idx_to_device()
        ratings_test = batch.get_ratings(is_test=True)

        model_predictions = model(ratings, idx, calculate_loss=False, mode=mode).cpu().detach().numpy()

        if not (input_data is output_data):
            model_predictions[batch.get_ratings().nonzero()] = -np.inf
        ndcg_dist.append(utils.NDCG_binary_at_k_batch(model_predictions, ratings_test))

    ndcg_dist = np.concatenate(ndcg_dist)
    return ndcg_dist[~np.isnan(ndcg_dist)].mean()


def run(model, data, batch_size, n_epochs, axis, beta, mode):
    global best_ranking_quality
    global ndcgs_tr_pr, ndcgs_tr_mf, ndcgs_va_pr, ndcgs_va_mf

    for epoch in range(n_epochs):
        model.train()
        NLL_loss = 0
        KLD_loss = 0

        for batch in batch_data_sampler(batch_size=batch_size, device=device, axis=axis, input_data=data, shuffle=True):
            ratings = batch.get_ratings_to_device()
            idx = batch.get_idx_to_device()

            for optimizer in model.opts:
                optimizer.zero_grad()

            (NLL, KLD), loss = model(ratings, idx, beta=beta, mode=mode)
            loss.backward()

            for optimizer in model.opts:
                optimizer.step()

            NLL_loss += NLL.item()
            KLD_loss += KLD.item()

        print('NLL_loss', NLL_loss, 'KLD_loss', KLD_loss)


def display_results_graph(best_ranking_quality, ndcgs_va_pr, ndcgs_tr_pr):
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


# ______________________________  Main Code _________________________________________________


device = torch.device("cpu")
init_seed()

# Load Data
data = (x.astype('float32') for x in utils.get_data(global_indexing=False, dataset='pro_sg'))
# for validation and testing 20% of rating data used to compare with prediction
train_data, valid_input_data, valid_output_data, test_input_data, test_output_data = data
n_users, n_items = train_data.shape
print(n_items, n_users)

# best_ranking_quality = information retrieval metrics for ranking quality: Recall@k and NDCG@k
# they used ndcg
best_ranking_quality = -np.inf
ndcgs_tr_pr, ndcgs_tr_mf, ndcgs_va_pr, ndcgs_va_mf = [], [], [], []
var_param_distance = []

# Set Model
model_params = get_model_params()
current_model = VAE(model_params, input_dim=(n_users, n_items), axis='users', device=device).to(device)
current_model.set_embeddings(train_data)
print(current_model)


# optimizer_encoder_i, optimizer_decoder_i, _ = get_opts(current_model)

best_model = current_model
for epoch in range(50):
    # Train
    run(current_model, train_data, batch_size=500, n_epochs=3, axis='users', mode='pr', beta=0.005)
    current_model.set_embeddings(train_data)
    run(current_model, train_data, batch_size=500, n_epochs=1, axis='users', mode='mf', beta=None)

    axis = 'users'
    ndcg_ = validate(current_model, train_data, train_data, axis, 'mf', 0.01)
    ndcgs_tr_mf.append(ndcg_)
    ndcg_ = validate(current_model, train_data, train_data, axis, 'pr', 0.01)
    ndcgs_tr_pr.append(ndcg_)
    ndcg_ = validate(current_model, valid_input_data, valid_output_data, axis, 'pr', 1)
    ndcgs_va_pr.append(ndcg_)

    # wait for graph to close
    clear_output(False)

    # compare model to best one seen so far
    if ndcg_ > best_ranking_quality:
        best_ranking_quality = ndcg_
        best_model = current_model
        # torch.save(model.state_dict(), ser_model_fn)

    # check stop criteria
    if ndcg_ < best_ranking_quality / 2 and epoch > 10:
        break

# model_i.load_state_dict(torch.load(ser_model_fn))
current_model = best_model

# Test
batch_size_test = 2000
current_model.eval()
n100_list, r20_list, r50_list = [], [], []

for batch in batch_data_sampler(batch_size=batch_size_test, device=device, axis='users', input_data=test_input_data, output_data=test_output_data,
                                samples_perc_per_epoch=1):
    user_ratings = batch.get_ratings_to_device()
    users_idx = batch.get_idx_to_device()
    user_ratings_test = batch.get_ratings(is_test=True)

    pred_val = current_model(user_ratings, users_idx, calculate_loss=False, mode='mf').cpu().detach().numpy()
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
