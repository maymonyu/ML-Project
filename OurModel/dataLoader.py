from collections import defaultdict
from scipy import sparse
# import scipy
import pandas as pd
import numpy as np
from copy import copy
import os


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def split_train_val_test_proportion(data, val_prop=0.1,test_prop=0.1):
    data_grouped_by_user = data.groupby('user')
    tr_list, val_list, te_list = list(), list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            test_idx = np.zeros(n_items_u, dtype='bool')
            test_idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True
            val_idx = np.zeros(n_items_u, dtype='bool')
            val_idx[np.random.choice(n_items_u, size=int(val_prop * n_items_u), replace=False).astype('int64')] = True
            tr_list.append(group[np.logical_and(np.logical_not(test_idx), np.logical_not(val_idx))])
            te_list.append(group[test_idx])
            val_list.append(group[val_idx])
        else:
            tr_list.append(group)

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)
    data_val = pd.concat(val_list)

    return data_tr, data_val, data_te


def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('user')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)

        if n_items_u >= 5:
            test_idx = np.zeros(n_items_u, dtype='bool')
            test_idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True
            tr_list.append(group[np.logical_not(test_idx)])
            te_list.append(group[test_idx])
        else:
            tr_list.append(group)

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def numerize(tp, show2id, profile2id):
    uid = map(lambda x: profile2id[x], tp['user'])
    sid = map(lambda x: show2id[x], tp['item'])
    return pd.DataFrame(data={'user': list(uid), 'item': list(sid)}, columns=['user', 'item'])


def data_partition(data_filename):
    DATA_DIR = './data/'
    data_table = pd.read_csv(DATA_DIR + data_filename + '.csv')
    pro_dir = os.path.join(DATA_DIR, 'processed')

    user_activity = get_count(data_table, 'user')

    unique_uid = user_activity.index
    unique_sid = pd.unique(data_table['item'])

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    train_data, val_data, test_data = split_train_val_test_proportion(data_table, 0.1, 0.1)
    val_tr_data, val_te_data = split_train_test_proportion(val_data)
    test_tr_data, test_te_data = split_train_test_proportion(test_data)

    print(get_count(data_table, 'user'))
    print(get_count(train_data, 'user'))
    print(get_count(val_data, 'user'))
    print(get_count(test_data, 'user'))

    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)

    train_data = numerize(train_data, show2id, profile2id)
    train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

    val_tr_data = numerize(val_tr_data, show2id, profile2id)
    val_tr_data.to_csv(os.path.join(pro_dir, 'validation_input.csv'), index=False)
    val_te_data = numerize(val_te_data, show2id, profile2id)
    val_te_data.to_csv(os.path.join(pro_dir, 'validation_output.csv'), index=False)

    test_tr_data = numerize(test_tr_data, show2id, profile2id)
    test_tr_data.to_csv(os.path.join(pro_dir, 'test_input.csv'), index=False)
    test_te_data = numerize(test_te_data, show2id, profile2id)
    test_te_data.to_csv(os.path.join(pro_dir, 'test_output.csv'), index=False)

    return len(unique_uid), len(unique_sid)


def load_train_data(csv_file, n_items, n_users, global_indexing=False):
    tp = pd.read_csv(csv_file)

    n_users = n_users if global_indexing else tp['user'].max() + 1

    rows, cols = tp['user'], tp['item']
    data = sparse.csr_matrix((np.ones_like(rows),
                              (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    print(data[0])
    return data


def load_tr_te_data(csv_file_tr, csv_file_te, n_items, n_users):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['user'].min(), tp_te['user'].min())
    end_idx = max(tp_tr['user'].max(), tp_te['user'].max())

    rows_tr, cols_tr = tp_tr['user'] - start_idx, tp_tr['item']
    rows_te, cols_te = tp_te['user'] - start_idx, tp_te['item']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                 (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                 (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te


def data_loader():
    DATA_DIR = './data/'
    pro_dir = os.path.join(DATA_DIR, 'processed')

    # run this line for the first time
    n_users, n_items = data_partition('ml-1m')

    train_data = load_train_data(os.path.join(pro_dir, 'train.csv'), n_items, n_users)
    validation_tr_data, validation_te_data = load_tr_te_data(
        os.path.join(pro_dir, 'validation_input.csv'), os.path.join(pro_dir, 'validation_output.csv'),
        n_items, n_users)
    test_tr_data, test_te_data = load_tr_te_data(
        os.path.join(pro_dir, 'test_input.csv'), os.path.join(pro_dir, 'test_output.csv'),
        n_items, n_users)
    return train_data, validation_tr_data, validation_te_data, test_tr_data, test_te_data, n_users, n_items
