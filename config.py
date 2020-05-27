#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#pylint: disable=C0301,C0326
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_file', type=str, default='model.pth.tar', help='File to save model.')
    parser.add_argument('--model_dir',  type=str, default='models', help='Directory for storing checkpoint file.')
    parser.add_argument('--resume',     type=str, default='', metavar='PATH', help='path to resume checkpoint (default: none)')
    parser.add_argument('--mode',       type=str, default='train', choices=('train', 'test'), help='train or test.')
    parser.add_argument('--log_dir',    type=str, default='runs_attn', help='Directory for logging.')
    parser.add_argument('--gpu',        type=str, default='0', help='Number of gpu to use')
    parser.add_argument('--seed',       type=int, default=1234, help='Random seed')

    # hyper parameter for local data and training
    parser.add_argument('--distr_type',   type=str,   default='uniform', choices=('uniform', 'dirichlet'), help='Distribution to construct local data.')
    parser.add_argument('--alpha',        type=float, default=1., help='alpha for dirichlet distribution. Must > 0 if dirichlet distribution is chosen.')
    parser.add_argument('--lr',           type=float, default=5e-3, help='learning rate.')
    parser.add_argument('--min_lr',       type=float, default=1e-4, help='minimum learning rate.')
    parser.add_argument('--decay_rate',   type=float, default=0.99, help='lr decay rate.')
    parser.add_argument('--batch_size',   type=int,   default=64, help='Batch size. (B)')
    parser.add_argument('--local_epochs', type=int,   default=2, help='Number of epoch in local. (E)')
    parser.add_argument('--num_workers',  type=int,   default=0, help='number of workers to preprocess data, must be 0 for mp agents.')

    # hyper parameters for central server
    parser.add_argument('--num_locals',  type=int,   default=10, help='number of local agents.')
    parser.add_argument('--num_per_rnd', type=int,   default=2, help='number of local agents to train per round.')
    parser.add_argument('--rounds',      type=int,   default=500, help='number of communication rounds.')
    parser.add_argument('--sample_rate', type=float, default=-1., help='sample rate of central data.')
    parser.add_argument('--policy',      type=str,   default='avg', choices=('avg', 'ewc', 'mas'), help='Policy for estimating parameter importance.')
    parser.add_argument('--estimate_weights_in_center', action='store_true', help='Estimate parameter importance in central server.')

    # hyper parameters for ewc train
    parser.add_argument('--coe',      type=float, default=0.5, help='The coefficient for local additional constraint.')
    parser.add_argument('--interval', type=float, default=1, help='The interval for weight estimation.')

    args = parser.parse_args()
    return args
