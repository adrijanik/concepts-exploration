#!/usr/bin/env python

import torch
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, n_H=20, p=4, K=3):
        super().__init__()
        self.K = K
        self.n_H = n_H
        self.xh = nn.Linear(p, n_H)
        self.hy = nn.Linear(n_H, K)

    def forward(self, x):
        h = self.xh(x)
        return h, self.hy(h)


class Iris(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.Tensor(X)
        self.y = torch.LongTensor(y)

    def __getitem__(self, ix):
        return self.X[ix, :], self.y[ix]

    def __len__(self):
        return self.X.shape[0]


def train(model, iterator, optimizer, device):
    epoch_loss = 0
    model.train()

    for i, (x, y) in enumerate(iterator):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)

        h, logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return model, epoch_loss / len(iterator)
