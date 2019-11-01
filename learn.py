from tqdm import tqdm
import torch
from torch.autograd import Variable
from utils import argsort_torch, chol_nonzero
import pdb
import scipy.sparse as sparse
import numpy as np
import datetime
import math


def train(args, data_loader_train, data_loader_dev, data_loader_test, model, loss_fn, optimizer):
    for epoch in range(1, args.epochs + 1):
        args.current_epoch = epoch
        if args.train:
            print("Train")
            if args.no_grad:
                train_no_grad_epoch(args, data_loader_train, model)
            else:
                train_epoch(args, data_loader_train, model, loss_fn, optimizer)
            train_loss, train_accuracy = stat_collect(args, data_loader_train, model, loss_fn, "Statistics", False)
            print("Train loss = {:.6f}".format(train_loss))
            print("Train accuracy = {:.6f}".format(train_accuracy))
        if args.dev and (epoch - 1) % args.nnz_interval == 0:
            print("Dev")
            dev_loss, dev_accuracy, nonzero_increase = stat_collect(args, data_loader_dev, model, loss_fn, "Statistics")
            print("Dev loss = {:.6f}".format(dev_loss))
            print("Dev accuracy = {:.6f}".format(dev_accuracy))
            if args.compute_nonzero:
                print("Nonzero increase = {:.6f}".format(nonzero_increase))
    if args.test:
        print("Test")
        test_loss, test_accuracy, nonzero_increase = stat_collect(args, data_loader_test, model, loss_fn, "Statistics")
        print("Test loss = {:.6f}".format(test_loss))
        print("Test accuracy = {:.6f}".format(test_accuracy))
        if args.compute_nonzero:
            print("Nonzero increase = {:.6f}".format(nonzero_increase))


def train_no_grad_epoch(args, data_loader, model):
    print("Epoch ", args.current_epoch)
    model.train()
    data_iter = data_loader.__iter__()
    num_batches_per_epoch = min(args.max_batches_per_train_epoch, len(data_loader))
    tqdm_bar = tqdm(data_iter, total=num_batches_per_epoch)
    i = 0
    accuracy = 0
    sample_num = 0
    model.require_no_grad()
    x_save = []
    y_save = []
    output_save = []
    y_nnz = []
    output_nnz = []
    for batch in data_iter:
        x, y = batch['x'], batch['y']
        input_xs = []
        y_seqs = []
        for j in range(0, len(x)):
            input_xs.append(Variable(x[j].type(torch.int64)))
            y_seqs.append(y[j].cpu().numpy())
        with torch.no_grad():
            nonzero_output_x_list, nonzero_y_list, nonzero_increase_list, output_x_seq_list\
                = model.forward_no_grad_batch(input_xs, y_seqs)
        for j in range(0, len(x)):
            nonzero_output_x, nonzero_y, nonzero_increase, output_x_seq = \
                nonzero_output_x_list[j], nonzero_y_list[j], nonzero_increase_list[j], output_x_seq_list[j]
            y_seq = y_seqs[j]
            if nonzero_output_x < nonzero_y and args.save_improved:
                print(nonzero_output_x, " ", nonzero_y, " ", nonzero_increase)
                x_save.append(x[j].cpu().numpy())
                y_save.append(y_seq)
                output_save.append(output_x_seq)
                y_nnz.append(nonzero_y)
                output_nnz.append(nonzero_output_x)
            accuracy += sum(output_x_seq - y_seq == 0) / len(y_seq)
        sample_num += len(x)
        tqdm_bar.update()
        model.print_nonzero_increase_list()
        i = i + 1
        if i > num_batches_per_epoch:
            data_iter.__del__()
            break
    tqdm_bar.close()
    if args.save_improved:
        filename = args.output_prefix + str(datetime.datetime.now()) + '_train' + '.npz'
        if len(x_save) > 0:
            np.savez(filename, x=x_save, y=y_save, output=output_save, y_nnz=y_nnz, output_nnz=output_nnz)
    return accuracy / sample_num


def train_epoch(args, data_loader, model, loss_fn, optimizer):
    print("Epoch ", args.current_epoch)
    model.train()
    data_iter = data_loader.__iter__()
    num_batches_per_epoch = min(args.max_batches_per_train_epoch, len(data_loader))
    tqdm_bar = tqdm(data_iter, total=num_batches_per_epoch)
    i = 0
    for batch in data_iter:
        x, y = batch['x'], batch['y']
        loss = 0
        for j in range(0, len(x)):
            input_x = Variable(x[j].type(torch.int64))
            output_x = model.forward(input_x).float()
            y_scaled = y[j].float()
            y_scaled = (y_scaled - torch.min(y_scaled)) / (torch.max(y_scaled) - torch.min(y_scaled))
            loss += loss_fn(output_x.squeeze(-1), y_scaled)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        tqdm_bar.update()
        i = i + 1
        if i > num_batches_per_epoch:
            data_iter.__del__()
            break
    tqdm_bar.close()


def stat_collect(args, data_loader, model, loss_fn, info, nnz_flag=True):
    print(info)
    model.eval()
    data_iter = data_loader.__iter__()
    num_batches_per_epoch = min(args.max_batches_per_test_epoch, len(data_loader))
    tqdm_bar = tqdm(data_iter, total=num_batches_per_epoch)
    i = 0
    accuracy = 0
    loss = 0
    nonzero_increase = 0
    sample_num = 0
    x_save = []
    y_save = []
    output_save = []
    y_nnz = []
    output_nnz = []
    for batch in data_iter:
        x, y = batch['x'], batch['y']
        for j in range(0, len(x)):
            with torch.no_grad():
                input_x = Variable(x[j].type(torch.int64))
                output_x = model.forward(input_x).float()
                output_x_seq = argsort_torch(output_x, args)
                y_seq = y[j].cpu().numpy()
                if args.compute_nonzero and nnz_flag:
                    nonzero_output_x = chol_nonzero(input_x.cpu(), output_x_seq)
                    nonzero_y = chol_nonzero(input_x.cpu(), y_seq)
                    if nonzero_output_x < nonzero_y and args.save_improved:
                        print(nonzero_output_x, " ", nonzero_y, " ", (nonzero_y - nonzero_output_x) / nonzero_y)
                        x_save.append(x[j].cpu().numpy())
                        y_save.append(y_seq)
                        output_save.append(output_x_seq)
                        y_nnz.append(nonzero_y)
                        output_nnz.append(nonzero_output_x)
                    nonzero_increase += (nonzero_output_x - nonzero_y) / nonzero_y
            accuracy += sum(output_x_seq - y_seq == 0) / len(output_x)
            y_scaled = y[j].float()
            y_scaled = (y_scaled - torch.min(y_scaled)) / (torch.max(y_scaled) - torch.min(y_scaled))
            loss += loss_fn(output_x.squeeze(-1), y_scaled)
            sample_num += 1
        tqdm_bar.update()
        i = i + 1
        if i > num_batches_per_epoch:
            data_iter.__del__()
            break
    tqdm_bar.close()
    if args.save_improved:
        filename = args.output_prefix + str(datetime.datetime.now()) + '.npz'
        if len(x_save) > 0:
            np.savez(filename, x=x_save, y=y_save, output=output_save, y_nnz=y_nnz, output_nnz=output_nnz)
    if args.compute_nonzero and nnz_flag:
        return loss / sample_num, accuracy / sample_num, nonzero_increase / sample_num
    else:
        return loss / sample_num, accuracy / sample_num
