# -*- coding: utf-8 -*-
import os
import datetime
import copy
import collections
from queue import Queue
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter

def train_local_mp(net, local_epochs, rnd, q, policy, w_d):
    """For multiprocessing train"""
    print(f"=> Train begins: Round {rnd}")
    if policy in ['ewc', 'mas'] and w_d is None:
        w_d = net.estimate_weights(policy)

    net.global_model = copy.deepcopy(net.model.state_dict())
    for _ in range(local_epochs):
        net.train(w_d)
    print(f"=> Test begins: Round {rnd}")
    test_acc, test_loss = net.test()
    q.put((test_acc, test_loss))

def test_local_mp(net, q, i):
    """For multiprocessing test"""
    print(f"=> Clients {i} Test Begins.")
    test_acc, test_loss = net.test()
    q.put((test_acc, test_loss))

class Trainer(ABC):
    """Base Trainer Class"""
    def __init__(self, global_args):
        self.fine = global_args.fine
        self.num_locals = global_args.num_locals
        self.num_per_rnd = global_args.num_per_rnd
        self.local_epochs = global_args.local_epochs
        self.rounds = global_args.rounds
        self.policy = global_args.policy
        self.sample_rate = global_args.sample_rate
        self.interval = global_args.interval
        self.resume = global_args.resume
        self.log_dir = global_args.log_dir
        self.estimate_weights_in_center = global_args.estimate_weights_in_center

        self.device = torch.device('cuda:0')
        self.data_alloc = None
        self.writer = None
        self.global_agent = None
        self.nets_pool = None
        self.q = Queue()
        self.local_attn_weights = list()

        self.writer = None
        if global_args.mode == 'train':
            writer_dir = os.path.join(f'{self.log_dir}/{self.fine}_{self.policy}_{self.num_locals}',
                                      datetime.datetime.now().strftime('%b%d_%H-%M'))
            self.writer = SummaryWriter(writer_dir)

    def __del__(self):
        if self.writer is not None:
            self.writer.close()

    def init_local_models(self):
        # duplicate the global model to local nets
        global_state = self.global_agent.model.state_dict()
        for net in self.nets_pool:
            net.load_data(self.data_alloc)
            net.build_model()
            net.model.load_state_dict(global_state)
        print(f'=> {len(self.nets_pool)} local nets init done.')

    def model_aggregation_avg(self):
        # compute average of models
        print('=> model aggregation with policy (avg)')
        dict_new = collections.defaultdict(list)
        # num of train examples of each agent
        weights = list()
        for net in self.nets_pool[:self.num_per_rnd]:
            weights.append(net.num_train)
            for k, v in net.model.state_dict().items():
                dict_new[k].append(v)
        # normalization
        weights = torch.as_tensor(weights, dtype=torch.float, device=self.device)
        weights.div_(weights.sum())
        for k, v in dict_new.items():
            v = torch.stack(v)
            expected_shape = [self.num_per_rnd] + [1] * (v.dim() - 1)
            dict_new[k] = torch.sum(v.mul_(weights.reshape(expected_shape)), dim=0)
        return dict_new

    def update_global(self, rnd):
        dict_new = self.model_aggregation_avg()

        # update global model and test
        self.global_agent.model.load_state_dict(dict_new)
        self.global_agent.update_lr(rnd, self.writer)
        print(f"=> Global Test begins: Round {rnd}")
        global_acc, global_loss = self.global_agent.test(rnd)
        self.writer.add_scalar('global/accuracy', global_acc, rnd)
        self.writer.add_scalar('global/loss', global_loss, rnd)

        local_test = list()
        while not self.q.empty():
            local_test.append(self.q.get())
        local_acc, local_loss = np.mean(np.asarray(local_test), axis=0)
        local_acc_std, local_loss_std = np.std(np.asarray(local_test), axis=0)
        self.writer.add_scalar('local/accuracy', local_acc, rnd)
        self.writer.add_scalar('local/loss', local_loss, rnd)
        self.writer.add_scalar('local/accuracy_std', local_acc_std, rnd)
        self.writer.add_scalar('local/loss_std', local_loss_std, rnd)
        self.global_agent.maybe_save(rnd, local_acc)

    def test(self):
        print("=> Test begins.")
        self.global_agent.test()

    @abstractmethod
    def build_local_models(self, global_args):
        pass

    @abstractmethod
    def train(self):
        pass
