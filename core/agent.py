# -*- coding: utf-8 -*-
import os
import time
import copy
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
from prefetch_generator import BackgroundGenerator

from . import utils

class Agent(ABC):
    """Base Agent Class."""
    def __init__(self, global_args, subset, fine):
        self.best_acc = 0.
        self.test_acc = 0.
        self.test_loss = 0.
        self.subset = subset
        self.fine = fine
        self.num_train = 0
        self.device = torch.device('cuda:0')

        self.rounds = global_args.rounds
        self.coe = global_args.coe
        self.max_lr = global_args.lr
        self.lr = self.max_lr
        self.min_lr = global_args.min_lr
        self.decay_rate = global_args.decay_rate
        self.batch_size = global_args.batch_size
        self.num_workers = global_args.num_workers

        self.model_dir = global_args.model_dir

    @abstractmethod
    def load_data(self):
        print("=> loading data")
        self.data = None
        self.train_loader = None
        self.test_loader = None

    @abstractmethod
    def build_model(self):
        print("=> building model")
        self.model = None
        self.shadow = None
        self.criterion = None
        self.optimizer = None

    def resume_model(self, resume_path):
        """optionally resume from a checkpoint"""
        if resume_path:
            resume_path = f'{self.model_dir}/{resume_path}.pth.tar'
            if os.path.isfile(resume_path):
                print(f"=> loading checkpoint '{resume_path}'")
                checkpoint = torch.load(resume_path, map_location=self.device)
                self.best_acc = checkpoint['best_acc']
                self.model.load_state_dict(checkpoint['state_dict'])
                print(f"=> loaded checkpoint '{resume_path}' (Round {checkpoint['rnd']})")
                del checkpoint
            else:
                print(f"=> no checkpoint found at '{resume_path}'")

    def ewc(self, train_loader=None):
        if train_loader is None:
            train_loader = self.train_loader

        tmp_weights = dict()
        for k, p in self.model.named_parameters():
            tmp_weights[k] = torch.zeros_like(p)

        self.model.eval()
        num_examples = 0
        for image_batch, label_batch in BackgroundGenerator(train_loader):
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            num_examples += image_batch.size(0)

            # compute output
            output, _ = self.model(image_batch)
            loss = self.criterion(output, label_batch)

            # compute gradient
            self.optimizer.zero_grad()
            loss.backward()
            for k, p in self.model.named_parameters():
                tmp_weights[k].add_(p.grad.detach() ** 2)

        for k, v in tmp_weights.items():
            tmp_weights[k] = torch.sum(v).div(num_examples)

        return tmp_weights

    def mas(self, train_loader=None):
        if train_loader is None:
            train_loader = self.train_loader

        tmp_weights = dict()
        for k, p in self.model.named_parameters():
            tmp_weights[k] = torch.zeros_like(p)

        self.model.eval()
        criterion = nn.MSELoss().cuda()
        num_examples = 0
        for image_batch, label_batch in BackgroundGenerator(train_loader):
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            num_examples += image_batch.size(0)

            # compute output
            output, *_ = self.model(image_batch)
            label_zero = torch.zeros_like(output)
            loss = criterion(output, label_zero)

            # compute gradient
            self.optimizer.zero_grad()
            loss.backward()
            for k, p in self.model.named_parameters():
                tmp_weights[k].add_(p.grad.detach().abs())

        # normalize
        for k, v in tmp_weights.items():
            tmp_weights[k] = torch.sum(v).div(num_examples)

        return tmp_weights

    def estimate_weights(self, policy, train_loader=None):
        print('=> estimating weights')
        if policy == 'ewc':
            return self.ewc(train_loader)
        elif policy == 'mas':
            return self.mas(train_loader)
        else:
            raise ValueError(f'Unsupported policy for estimating weights: {policy}.')

    def test(self, rnd=None, writer=None):
        batch_time = utils.AverageMeter()
        loss_meter = utils.AverageMeter()
        top1 = utils.AverageMeter()
        # switch to eval mode
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for image_batch, label_batch in BackgroundGenerator(self.test_loader):
                image_batch, label_batch = image_batch.to(self.device), label_batch.to(self.device)

                # compute output
                output, _ = self.model(image_batch)
                loss = self.criterion(output, label_batch)
                loss_meter.update(loss.item(), label_batch.size(0))

                # measure accuracy
                acc, *_ = utils.accuracy(output, label_batch)
                top1.update(acc[0].item(), label_batch.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

            print(f' * Accuracy {top1.avg:.3f}')
            if rnd and writer is not None:
                writer.add_scalar('global/loss', loss_meter.avg, rnd)
                writer.add_scalar('global/accuracy', top1.avg, rnd)

        self.test_acc = top1.avg
        self.test_loss = loss_meter.avg
        return self.test_acc, self.test_loss

    def train(self, w_d):
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        ce_loss_meter = utils.AverageMeter()
        cons_loss_meter = utils.AverageMeter()
        loss_meter = utils.AverageMeter()
        top1 = utils.AverageMeter()
        # switch to train mode
        self.model.train()

        end = time.time()
        for image_batch, label_batch in BackgroundGenerator(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            image_batch, label_batch = image_batch.to(self.device), label_batch.to(self.device)

            # compute output and loss
            output, _ = self.model(image_batch)
            ce_loss = self.criterion(output, label_batch)
            ce_loss_meter.update(ce_loss.item(), label_batch.size(0))
            cons_loss = 0
            if w_d is not None:
                for k, v in self.model.named_parameters():
                    cons_loss += w_d[k] * torch.sum((self.global_model[k] - v) ** 2)
                cons_loss *= self.coe
                cons_loss_meter.update(cons_loss.item(), label_batch.size(0))
            loss = ce_loss + cons_loss
            loss_meter.update(loss.item(), label_batch.size(0))

            # measure accuracy
            acc, *_ = utils.accuracy(output, label_batch)
            top1.update(acc[0].item(), label_batch.size(0))

            # compute gradient and update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    def set_lr(self, lr):
        self.lr = lr
        for param in self.optimizer.param_groups:
            param['lr'] = self.lr

    def update_lr(self, rnd, writer=None):
        if writer is not None:
            writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], rnd)
        if self.lr > self.min_lr:
            # exponential
            # self.lr *= self.decay_rate
            # cosine
            self.lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(rnd * np.pi / self.rounds))
        for param in self.optimizer.param_groups:
            param['lr'] = self.lr

    def maybe_save(self, rnd, local_acc):
        is_best = local_acc > self.best_acc
        if is_best:
            self.best_acc = local_acc

        utils.save_checkpoint({
            'rnd': rnd + 1,
            'state_dict': self.model.state_dict(),
            'best_acc': self.best_acc,
            # 'optimizer' : self.optimizer.state_dict(),
        }, is_best, model_dir=self.model_dir, prefix=self.fine)
