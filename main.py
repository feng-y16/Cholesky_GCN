import numpy as np
import matplotlib as plt
import pdb
import scipy
import torch
from torch.utils.data import DataLoader
from dataset import get_data_loader
from parsing import parse_args
from utils import collate_fn, rearrange
from model import Net
from learn import train
import multiprocessing as mp
import math


def loss_fn(output, label):
    loss = torch.abs(output - label)
    mask = torch.lt(loss, 1e-5)
    loss[mask] = 0
    loss = loss ** 0.01
    weight = torch.linspace(2, 1e-3, len(loss)).cuda()
    loss = loss * weight
    return torch.mean(loss)


def main():
    mp.set_start_method('spawn')
    args = parse_args()
    torch.set_grad_enabled(True)
    data_loader_train, data_loader_dev, data_loader_test = get_data_loader(args)
    print("Data loaded.")
    model = Net(args).to(args.device)
    print("Model set.")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("Optimizer set.")
    train(args, data_loader_train, data_loader_dev, data_loader_test, model, loss_fn, optimizer)


def model_test():
    args = parse_args()
    torch.set_grad_enabled(True)
    args.dropout = 0
    args.cuda = False
    dataset = DatasetBase(args)
    print("Data loaded.")
    x, y = dataset.__getitem__(0)
    model = Net(args)
    result1 = model.forward(x.type(torch.int64))
    result2 = model.forward(rearrange(x, 0, 1).type(torch.int64))
    return result1, result2


if __name__ == "__main__":
    # model_test()
    main()
