import argparse
import os
import torch
import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='GCN model')
    # setup
    parser.add_argument('--train', action='store_true', default=False, help='Whether or not to train model')
    parser.add_argument('--dev', action='store_true', default=False, help='Whether or not to run model on dev set')
    parser.add_argument('--test', action='store_true', default=False, help='Whether or not to run model on test set')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of train data')
    parser.add_argument('--dev_ratio', type=float, default=0.1, help='Ratio of dev data')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Ratio of test data')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--compute_nonzero', action='store_true', default=False, help='Whether or not to compute '
                                                                                      'increase in nonzero elements')
    parser.add_argument('--nnz_interval', type=int, default=1, help='How many train epoch to be with a dev epoch')
    parser.add_argument('--save_improved', action='store_true', default=False, help='Whether or not to save '
                                                                                    'improved data')
    parser.add_argument('--dataset_path', default='data/', help='Path of dataset')
    parser.add_argument('--dataset_file', default='small_case', help='File name of dataset')
    parser.add_argument('--feature_dim', type=int, default=1, help='Feature dim for gcn models. [default:512]')
    # learning
    parser.add_argument('--optimizer', type=str, default="adam", help='optimizer to use [default: adam]')
    parser.add_argument('--init_lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum to use with SGD')
    parser.add_argument('--lr_decay', type=float, default=1, help='learning rate decay[default: 1]')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 Regularization penalty [default: 0]')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs for train [default: 20]')
    parser.add_argument('--max_batches_per_train_epoch', type=int, default=10000, help='max batches to per train epoch')
    parser.add_argument('--max_batches_per_test_epoch', type=int, default=10000, help='max batches to per test epoch')
    parser.add_argument('--batch_size_train', type=int, default=32, help='batch size for training')
    parser.add_argument('--batch_size_dev', type=int, default=32, help='batch size for dev')
    parser.add_argument('--batch_size_test', type=int, default=32, help='batch size for testing')
    parser.add_argument('--batch_splits', type=int, default=1, help='Splits batch size into smaller batches')
    parser.add_argument('--dropout', type=float, default=0, help='Amount of dropout to apply on last hidden layer')
    parser.add_argument('--save_dir', type=str, default='saves/', help='where to dump the model')
    parser.add_argument('--results_path', type=str, default='logs/', help='where to save the result logs')
    parser.add_argument('--no_grad', action='store_true', default=False, help='whether or not to use the gradient')
    # device
    parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu')

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'
    args.current_epoch = None
    args.lr = args.init_lr
    args.dataset_path = os.path.dirname(os.path.realpath(__file__)) + "/" + args.dataset_path + args.dataset_file
    args.dataset_path += '.npz'
    args.output_prefix = os.path.dirname(os.path.realpath(__file__)) + "/output/" + args.dataset_file
    return args
