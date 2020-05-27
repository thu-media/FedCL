# -*- coding: utf-8 -*-
from itertools import chain
import copy
import torch
from torch import nn
import torch.nn.functional as F

class CifarModel(nn.Module):

    def __init__(self, num_class=10):
        super().__init__()

        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            # fc3
            nn.Linear(64 * 8 * 8, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            # fc4
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

            # softmax
            nn.Linear(192, num_class),
        )
        for layer in chain(self.features, self.classifier):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = self.features(x)
        out = x.view(x.size(0), -1)
        out = self.classifier(out)
        return out, x
