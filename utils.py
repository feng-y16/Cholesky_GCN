import torch
import numpy as np
from copy import deepcopy
import scipy.sparse as sparse
from scipy.linalg import cholesky
import pdb


def dataset_index_generator(total, train_ratio, dev_ratio, test_ratio, random=False):
    if random:
        index = torch.randperm(total).numpy()
    else:
        index = np.array(range(0, total))
    dev_ratio += train_ratio
    test_ratio += dev_ratio
    train_index = index[0: round(total * train_ratio)]
    dev_index = index[round(total * train_ratio): round(total * dev_ratio)]
    test_index = index[round(total * dev_ratio): round(total * test_ratio)]
    return train_index, dev_index, test_index


def collate_fn(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    return {
        'x': x,
        'y': y,
    }


def argsort_torch(x, args):
    x = x.squeeze(-1)
    if args.cuda:
        return np.argsort(x.cpu().detach().numpy())
    else:
        return np.argsort(x.detach().numpy())


def rearrange(x, i, j):
    size_matrix = torch.max(x) + 1
    row = x[0, :].squeeze(-1).numpy()
    col = x[1, :].squeeze(-1).numpy()
    data = np.ones(np.size(row))
    matrix = sparse.coo_matrix((data, (row, col)), shape=(size_matrix, size_matrix)).toarray()
    matrix[[i, j], :] = matrix[[j, i], :]
    matrix[:, [i, j]] = matrix[:, [j, i]]
    return torch.from_numpy(np.vstack((sparse.find(matrix)[0], sparse.find(matrix)[1])))


def chol_nonzero(x, seq):
    size_matrix = torch.max(x) + 1
    row = x[0, :].squeeze(-1).numpy()
    col = x[1, :].squeeze(-1).numpy()
    data = np.ones(np.size(row))
    matrix = sparse.coo_matrix((data, (row, col)), shape=(size_matrix, size_matrix)).toarray()
    matrix = matrix + np.transpose(matrix)
    matrix = matrix[seq]
    matrix = matrix[:, seq]
    for i in range(0, size_matrix):
        matrix[i, i] += 2 * size_matrix
    matrix = cholesky(matrix)
    return len(np.nonzero(matrix)[0])


def tensor_list_add(list1, list2):
    result = []
    for i in range(0, len(list1)):
        result.append(list1[i] + list2[i])
    return result


def tensor_list_minus(list1, list2):
    result = []
    for i in range(0, len(list1)):
        result.append(list1[i] - list2[i])
    return result


def tensor_list_multiply(list1, multiplier):
    result = []
    for i in range(0, len(list1)):
        result.append(list1[i] * multiplier)
    return result


def tensor_list_linear(list1, multiplier1, list2, multiplier2):
    result = []
    for i in range(0, len(list1)):
        result.append(list1[i] * multiplier1 + list2[i] * multiplier2)
    return result
