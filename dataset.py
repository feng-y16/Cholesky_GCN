import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import scipy.sparse as sparse
from utils import collate_fn, dataset_index_generator


def load_dataset(path):
    dataset_temp = np.load(path, encoding="latin1", allow_pickle=True)
    x_temp = dataset_temp['x']
    y_temp = dataset_temp['y']
    return x_temp, y_temp


class DatasetBase(Dataset):
    def __init__(self, args):
        self.args = args
        self.x_data, self.y_data = load_dataset(args.dataset_path)
        self.len = np.size(self.y_data)

    def __getitem__(self, index):
        x = self.x_data[index]
        x = torch.from_numpy(np.vstack((sparse.find(x)[0], sparse.find(x)[1])))
        y = torch.from_numpy(np.array(self.y_data[index]))
        y = y.float()
        y = (y - torch.min(y)) / (torch.max(y) - torch.min(y) + 1e-15)
        return x.to(self.args.device), y.to(self.args.device)

    def __len__(self):
        return self.len


class DatasetGeneral(Dataset):
    def __init__(self, args, x_data, y_data, index):
        self.args = args
        self.x_data, self.y_data = x_data, y_data
        self.index = index
        self.len = np.size(index)

    def __getitem__(self, index):
        x = self.x_data[self.index[index]]
        x = torch.from_numpy(np.vstack((sparse.find(x)[0], sparse.find(x)[1])))
        y = torch.from_numpy(np.array(self.y_data[self.index[index]]))
        return x.to(self.args.device), y.to(self.args.device)

    def __len__(self):
        return self.len


def get_data_loader(args):
    dataset = DatasetBase(args)
    train_index, dev_index, test_index = dataset_index_generator(dataset.__len__(), args.train_ratio, args.dev_ratio,
                                                                 args.test_ratio, False)
    dataset_train = DatasetGeneral(args, dataset.x_data, dataset.y_data, train_index)
    dataset_dev = DatasetGeneral(args, dataset.x_data, dataset.y_data, dev_index)
    dataset_test = DatasetGeneral(args, dataset.x_data, dataset.y_data, test_index)
    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size_train, shuffle=True,
                                   num_workers=args.num_workers, drop_last=False, collate_fn=collate_fn)
    data_loader_dev = DataLoader(dataset_dev, batch_size=args.batch_size_dev, shuffle=True,
                                 num_workers=args.num_workers, drop_last=False, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size_test, shuffle=True,
                                  num_workers=args.num_workers, drop_last=False, collate_fn=collate_fn)
    return data_loader_train, data_loader_dev, data_loader_test
