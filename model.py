import torch
from torch_geometric.nn import GCNConv, ChebConv
from torch.autograd import Variable
import torch.nn.functional as functional
from utils import tensor_list_add, tensor_list_minus, tensor_list_multiply, tensor_list_linear,\
    argsort_torch, chol_nonzero
import numpy as np
import copy
import time
import pdb


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        increase_multiplier = 5
        self.layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6']
        self.saved_parameters = {'parameters': [], 'nonzero_increase': []}
        self.conv1 = GCNConv(args.feature_dim, increase_multiplier ** 2, cached=False)
        self.conv2 = GCNConv(increase_multiplier ** 2, increase_multiplier ** 3, cached=False)
        self.conv3 = GCNConv(increase_multiplier**3, increase_multiplier**3, cached=False)
        self.conv4 = GCNConv(increase_multiplier ** 3, increase_multiplier ** 3, cached=False)
        self.conv5 = GCNConv(increase_multiplier ** 3, increase_multiplier ** 3, cached=False)
        self.conv6 = GCNConv(increase_multiplier**3, 1, cached=False)

    def forward(self, edge_index):
        x = torch.ones(torch.max(edge_index) + 1, 1).to(self.args.device)
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)
        x = self.conv6(x, edge_index)
        x = functional.dropout(x, p=self.args.dropout, training=self.training)
        x = (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-15)
        return x

    def get_parameters(self):
        parameters = []
        for layer in self.layer_names:
            command = 'parameters.append(self.' + layer + '.weight)'
            exec(command)
        return parameters

    def require_no_grad(self):
        i = 0
        for layer in self.layer_names:
            command = 'self.' + layer + '.requires_grad = False'
            exec(command)
            i += 1
        return

    def change_parameters(self, new_parameters):
        i = 0
        for layer in self.layer_names:
            parameters_layer = torch.nn.Parameter(new_parameters[i])
            parameters_layer.requires_grad = False
            command = 'self.' + layer + '.weight = parameters_layer'
            exec(command)
            i += 1
        return

    def rand_parameters(self, mu, sigma):
        parameters = []
        for layer in self.layer_names:
            loc = locals()
            command = 'size = self.' + layer + '.weight.size()'
            exec(command)
            size = loc['size']
            parameters.append(mu + sigma * torch.randn(size))
        return parameters

    def get_non_zeros(self, edge_index, y_seq):
        output_x = self.forward(edge_index).float()
        output_x_seq = argsort_torch(output_x, self.args)
        nonzero_output_x = chol_nonzero(edge_index.cpu(), output_x_seq)
        nonzero_y = chol_nonzero(edge_index.cpu(), y_seq)
        nonzero_increase = (nonzero_output_x - nonzero_y) / nonzero_y
        return nonzero_output_x, nonzero_y, nonzero_increase, output_x_seq

    def get_non_zeros_batch(self, edge_indexes, y_seqs):
        output_x_seq_list = []
        nonzero_output_x_list = []
        nonzero_y_list = []
        nonzero_increase_list = []
        for i in range(0, len(y_seqs)):
            nonzero_output_x, nonzero_y, nonzero_increase, output_x_seq = self.get_non_zeros(edge_indexes[i], y_seqs[i])
            nonzero_output_x_list.append(nonzero_output_x)
            nonzero_y_list.append(nonzero_y)
            nonzero_increase_list.append(nonzero_increase)
            output_x_seq_list.append(output_x_seq)
        return nonzero_output_x_list, nonzero_y_list, nonzero_increase_list, output_x_seq_list

    def get_increase_mean_tails(self):
        increase_list = self.saved_parameters['nonzero_increase'].copy()
        increase_list = list(np.argsort(np.array(increase_list)))
        length = len(increase_list)
        return increase_list.index(0), increase_list.index(length - 1), increase_list.index(length - 2)

    def get_parameters_list_mean(self):
        mean_parameters = self.saved_parameters['parameters'][0]
        for i in range(1, len(self.saved_parameters['parameters'])):
            mean_parameters = tensor_list_add(mean_parameters, self.saved_parameters['parameters'][i])
        return tensor_list_multiply(mean_parameters, 1.0/len(self.saved_parameters['parameters']))

    def print_nonzero_increase_list(self):
        print_list = [round(i, 4) for i in self.saved_parameters['nonzero_increase']]
        print_list.sort()
        print(print_list, flush=True)
        return

    def forward_no_grad(self, edge_index, y_seq):
        if len(self.saved_parameters['nonzero_increase']) < 3:
            nonzero_output_x, nonzero_y, nonzero_increase, output_x_seq = self.get_non_zeros(edge_index, y_seq)
            self.saved_parameters['parameters'].append(self.get_parameters())
            self.saved_parameters['nonzero_increase'].append(nonzero_increase)
            self.change_parameters(self.rand_parameters(0, 1))
            return nonzero_output_x, nonzero_y, nonzero_increase, output_x_seq
        else:
            increase_list = self.saved_parameters['nonzero_increase']
            mean_parameters = self.get_parameters_list_mean()
            min_increase, max_increase, second_max_increase = self.get_increase_mean_tails()
            mirror_parameters = tensor_list_multiply(mean_parameters, 2)
            mirror_parameters = tensor_list_minus(mirror_parameters, self.saved_parameters['parameters'][max_increase])
            self.change_parameters(mirror_parameters)
            nonzero_output_x, nonzero_y, nonzero_increase, output_x_seq = self.get_non_zeros(edge_index, y_seq)
            if increase_list[min_increase] <= nonzero_increase <= increase_list[second_max_increase]:
                self.saved_parameters['parameters'].append(self.get_parameters())
                self.saved_parameters['nonzero_increase'].append(nonzero_increase)
                return nonzero_output_x, nonzero_y, nonzero_increase, output_x_seq
            elif nonzero_increase < increase_list[min_increase]:
                mirror_parameters_2 = tensor_list_multiply(mean_parameters, 3)
                mirror_parameters_2 = tensor_list_minus(
                    mirror_parameters_2, tensor_list_multiply(self.saved_parameters['parameters'][max_increase], 2))
                self.change_parameters(mirror_parameters_2)
                nonzero_output_x_2, nonzero_y_2, nonzero_increase_2, output_x_seq_2 \
                    = self.get_non_zeros(edge_index, y_seq)
                if nonzero_increase_2 < nonzero_increase:
                    self.saved_parameters['parameters'].append(self.get_parameters())
                    self.saved_parameters['nonzero_increase'].append(nonzero_increase_2)
                    return nonzero_output_x_2, nonzero_y_2, nonzero_increase_2, output_x_seq_2
                else:
                    self.change_parameters(mirror_parameters)
                    self.saved_parameters['parameters'].append(self.get_parameters())
                    self.saved_parameters['nonzero_increase'].append(nonzero_increase)
                    return nonzero_output_x, nonzero_y, nonzero_increase, output_x_seq
            if increase_list[second_max_increase] < nonzero_increase <= increase_list[max_increase]:
                mirror_parameters_2 = tensor_list_multiply(mean_parameters, 1.5)
                mirror_parameters_2 = tensor_list_minus(
                    mirror_parameters_2, tensor_list_multiply(self.saved_parameters['parameters'][max_increase], 0.5))
                self.change_parameters(mirror_parameters_2)
                nonzero_output_x_2, nonzero_y_2, nonzero_increase_2, output_x_seq_2 \
                    = self.get_non_zeros(edge_index, y_seq)
                if nonzero_increase_2 < nonzero_increase:
                    self.saved_parameters['parameters'].append(self.get_parameters())
                    self.saved_parameters['nonzero_increase'].append(nonzero_increase_2)
                    return nonzero_output_x_2, nonzero_y_2, nonzero_increase_2, output_x_seq_2
            if nonzero_increase > increase_list[max_increase]:
                mirror_parameters_2 = tensor_list_multiply(mean_parameters, 0.5)
                mirror_parameters_2 = tensor_list_add(
                    mirror_parameters_2, tensor_list_multiply(self.saved_parameters['parameters'][max_increase], 0.5))
                self.change_parameters(mirror_parameters_2)
                nonzero_output_x_2, nonzero_y_2, nonzero_increase_2, output_x_seq_2 \
                    = self.get_non_zeros(edge_index, y_seq)
                if nonzero_increase_2 < nonzero_increase:
                    self.saved_parameters['parameters'].append(self.get_parameters())
                    self.saved_parameters['nonzero_increase'].append(nonzero_increase_2)
                    return nonzero_output_x_2, nonzero_y_2, nonzero_increase_2, output_x_seq_2
            shrink_t = 0.5
            mean_parameters_weighted = tensor_list_multiply(mean_parameters, 1 - shrink_t)
            for i in range(0, len(self.saved_parameters['parameters'])):
                self.saved_parameters['parameters'][i] = \
                    tensor_list_add(mean_parameters_weighted, tensor_list_multiply(
                        self.saved_parameters['parameters'][i], shrink_t))
                self.change_parameters(self.saved_parameters['parameters'][i])
                nonzero_output_x, nonzero_y, nonzero_increase, output_x_seq = self.get_non_zeros(edge_index, y_seq)
                self.saved_parameters['nonzero_increase'][i] = nonzero_increase
            return nonzero_output_x, nonzero_y, nonzero_increase, output_x_seq

    def forward_no_grad_batch(self, edge_indexes, y_seqs):
        if len(list(set(self.saved_parameters['nonzero_increase']))) < 3:
            nonzero_output_x, nonzero_y, nonzero_increase, output_x_seq = self.get_non_zeros_batch(edge_indexes, y_seqs)
            self.saved_parameters['parameters'].append(self.get_parameters())
            self.saved_parameters['nonzero_increase'].append(np.mean(np.array(nonzero_increase)))
            self.change_parameters(self.rand_parameters(0, 1))
            return nonzero_output_x, nonzero_y, nonzero_increase, output_x_seq
        else:
            increase_list = self.saved_parameters['nonzero_increase']
            mean_parameters = self.get_parameters_list_mean()
            min_increase, max_increase, second_max_increase = self.get_increase_mean_tails()
            mirror_parameters = tensor_list_multiply(mean_parameters, 2)
            mirror_parameters = tensor_list_minus(mirror_parameters, self.saved_parameters['parameters'][max_increase])
            self.change_parameters(mirror_parameters)
            nonzero_output_x, nonzero_y, nonzero_increase, output_x_seq = self.get_non_zeros_batch(edge_indexes, y_seqs)
            nonzero_increase_mean = np.mean(np.array(nonzero_increase))
            if increase_list[min_increase] <= nonzero_increase_mean <= increase_list[second_max_increase]:
                self.saved_parameters['parameters'].append(self.get_parameters())
                self.saved_parameters['nonzero_increase'].append(nonzero_increase_mean)
                return nonzero_output_x, nonzero_y, nonzero_increase, output_x_seq
            elif nonzero_increase_mean < increase_list[min_increase]:
                mirror_parameters_2 = tensor_list_multiply(mean_parameters, 3)
                mirror_parameters_2 = tensor_list_minus(
                    mirror_parameters_2, tensor_list_multiply(self.saved_parameters['parameters'][max_increase], 2))
                self.change_parameters(mirror_parameters_2)
                nonzero_output_x_2, nonzero_y_2, nonzero_increase_2, output_x_seq_2 \
                    = self.get_non_zeros_batch(edge_indexes, y_seqs)
                nonzero_increase_2_mean = np.mean(np.array(nonzero_increase_2))
                if nonzero_increase_2_mean < nonzero_increase_mean:
                    self.saved_parameters['parameters'].append(self.get_parameters())
                    self.saved_parameters['nonzero_increase'].append(nonzero_increase_2_mean)
                    return nonzero_output_x_2, nonzero_y_2, nonzero_increase_2, output_x_seq_2
                else:
                    self.change_parameters(mirror_parameters)
                    self.saved_parameters['parameters'].append(self.get_parameters())
                    self.saved_parameters['nonzero_increase'].append(nonzero_increase_mean)
                    return nonzero_output_x, nonzero_y, nonzero_increase, output_x_seq
            if increase_list[second_max_increase] < nonzero_increase_mean <= increase_list[max_increase]:
                mirror_parameters_2 = tensor_list_multiply(mean_parameters, 1.5)
                mirror_parameters_2 = tensor_list_minus(
                    mirror_parameters_2, tensor_list_multiply(self.saved_parameters['parameters'][max_increase], 0.5))
                self.change_parameters(mirror_parameters_2)
                nonzero_output_x_2, nonzero_y_2, nonzero_increase_2, output_x_seq_2 \
                    = self.get_non_zeros_batch(edge_indexes, y_seqs)
                nonzero_increase_2_mean = np.mean(np.array(nonzero_increase_2))
                if nonzero_increase_2_mean < nonzero_increase_mean:
                    self.saved_parameters['parameters'].append(self.get_parameters())
                    self.saved_parameters['nonzero_increase'].append(nonzero_increase_2_mean)
                    return nonzero_output_x_2, nonzero_y_2, nonzero_increase_2, output_x_seq_2
            if nonzero_increase_mean > increase_list[max_increase]:
                mirror_parameters_2 = tensor_list_multiply(mean_parameters, 0.5)
                mirror_parameters_2 = tensor_list_add(
                    mirror_parameters_2, tensor_list_multiply(self.saved_parameters['parameters'][max_increase], 0.5))
                self.change_parameters(mirror_parameters_2)
                nonzero_output_x_2, nonzero_y_2, nonzero_increase_2, output_x_seq_2 \
                    = self.get_non_zeros_batch(edge_indexes, y_seqs)
                nonzero_increase_2_mean = np.mean(np.array(nonzero_increase_2))
                if nonzero_increase_2_mean < nonzero_increase_mean:
                    self.saved_parameters['parameters'].append(self.get_parameters())
                    self.saved_parameters['nonzero_increase'].append(nonzero_increase_2_mean)
                    return nonzero_output_x_2, nonzero_y_2, nonzero_increase_2, output_x_seq_2
            shrink_t = 0.9
            mean_parameters_weighted = tensor_list_multiply(mean_parameters, 1 - shrink_t)
            for i in range(0, len(self.saved_parameters['parameters'])):
                self.saved_parameters['parameters'][i] = \
                    tensor_list_add(mean_parameters_weighted, tensor_list_multiply(
                        self.saved_parameters['parameters'][i], shrink_t))
                self.change_parameters(self.saved_parameters['parameters'][i])
                nonzero_output_x, nonzero_y, nonzero_increase, output_x_seq = \
                    self.get_non_zeros_batch(edge_indexes, y_seqs)
                nonzero_increase_mean = np.mean(np.array(nonzero_increase))
                self.saved_parameters['nonzero_increase'][i] = nonzero_increase_mean
            return nonzero_output_x, nonzero_y, nonzero_increase, output_x_seq
