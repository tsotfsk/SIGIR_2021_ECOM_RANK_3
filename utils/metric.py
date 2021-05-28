import numpy as np


def hit(pos_index, pos_len):

    result = np.cumsum(pos_index, axis=1)
    return (result > 0).astype(int)


def mrr(pos_index, pos_len):

    idxs = pos_index.argmax(axis=1)
    result = np.zeros_like(pos_index, dtype=np.float)
    for row, idx in enumerate(idxs):
        if pos_index[row, idx] > 0:
            result[row, idx:] = 1 / (idx + 1)
        else:
            result[row, idx:] = 0
    return result


def recall(pos_index, pos_len):

    return np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)


def ndcg(pos_index, pos_len):
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)
    iranks = np.zeros_like(pos_index, dtype=np.float)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]

    ranks = np.zeros_like(pos_index, dtype=np.float)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    result = dcg / idcg
    return result


def precision(pos_index, pos_len):

    return pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1]+1)


def f1(pos_index, pos_len):
    pre = precision(pos_index, pos_len)
    re = recall(pos_index, pos_len)
    f1 = 2 * pre * re / (pre + re + 1e-25)
    return f1


metric_dict = {
    'ndcg': ndcg,
    'hit': hit,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'mrr': mrr
}