#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch import nn
import torch.multiprocessing as mp

import config
from core import Agent, Trainer, train_local_mp
from model import MNISTModel
from data import MNISTData

class MNISTAgent(Agent):
    """MNISTAgent for MNIST and Fashion-MNIST."""
    def __init__(self, global_args, subset=tuple(range(10))):
        super().__init__(global_args, subset, fine='MNIST')
        self.distr_type = global_args.distr_type
        if self.distr_type == 'uniform':
            self.distribution = np.array([0.1] * 10)
        elif self.distr_type == 'dirichlet':
            self.distribution = np.random.dirichlet([global_args.alpha] * 10)
        else:
            raise ValueError(f'Invalid distribution type: {self.distr_type}.')
        self.subset = subset

    def load_data(self, data_alloc, center=False):
        print("=> loading data")
        if center:
            self.train_loader, self.test_loader, self.num_train = data_alloc.create_dataset_for_center(
                self.batch_size, self.num_workers)
        else:
            self.train_loader, self.test_loader, self.num_train = data_alloc.create_dataset_for_client(
                self.distribution, self.batch_size, self.num_workers, self.subset)

    def build_model(self):
        print("=> building model")
        self.model = MNISTModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                                         momentum=0.9, weight_decay=1e-4)


class MNISTTrainer(Trainer):
    """MNIST Trainer."""
    def __init__(self, global_args):
        super().__init__(global_args)
        self.data_alloc = MNISTData(self.num_locals, self.sample_rate)

        # init the global model
        self.global_agent = MNISTAgent(global_args)
        self.global_agent.load_data(self.data_alloc, center=True)
        self.global_agent.build_model()
        self.global_agent.resume_model(self.resume)

    def build_local_models(self, global_args):
        self.nets_pool = list()
        for _ in range(self.num_locals):
            self.nets_pool.append(MNISTAgent(global_args))
        self.init_local_models()

    def train(self):
        for rnd in range(self.rounds):
            np.random.shuffle(self.nets_pool)
            pool = mp.Pool(5)
            self.q = mp.Manager().Queue()
            dict_new = self.global_agent.model.state_dict()
            if self.estimate_weights_in_center and rnd % self.interval == 0:
                w_d = self.global_agent.estimate_weights(self.policy)
            else:
                w_d = None
            for net in self.nets_pool[:self.num_per_rnd]:
                net.model.load_state_dict(dict_new)
                net.set_lr(self.global_agent.lr)
                pool.apply_async(train_local_mp, (net, self.local_epochs, rnd, self.q, self.policy, w_d))
                # train_local_mp(net, self.local_epochs, rnd, self.q, self.policy, w_d)
            pool.close()
            pool.join()
            self.update_global(rnd)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    mp.set_start_method('forkserver')

    mnist_trainer = MNISTTrainer(args)

    # test
    if args.mode == 'test':
        mnist_trainer.test()
        return

    mnist_trainer.build_local_models(args)
    mnist_trainer.train()

if __name__ == '__main__':
    args = config.get_args()
    args.fine = 'MNIST'
    main()
