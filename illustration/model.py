#!/usr/bin/env python
import torch
from torch.utils.data import Dataset
from torch import nn


class MLP(nn.Module):
    """
    One layer MLP

    :param n_H: Number of hidden units.
    :param p: Number of input dimensions.
    :param K: Number of output classes.

    Examples
    --------
    >>> model = MLP()
    """
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
    """
    Torch Dataset from Pandas
    """
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.Tensor(X)
        self.y = torch.LongTensor(y)

    def __getitem__(self, ix):
        return self.X[ix, :], self.y[ix]

    def __len__(self):
        return self.X.shape[0]


def train(model, iterator, optimizer, device):
    """
    Train for one Epoch

    :param model: A nn.Module object.
    :param iterator: A DataLoader with (x, y) tuples for data.
    :param optimizer: The thing that updates the parameters. From the
      torch.optim library.
    :param device: CPU or GPU?

    :return: A tuple with two elements
       - model: The model with updated parameters.
       - avg_loss: A float giving the epoch's average loss

    Examples
    --------
    >>> iris = datasets.load_iris()
    >>> data = Iris(iris.data, iris.target)
    >>> model = MLP()
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    >>> iterator = DataLoader(data, 32)
    >>> train(model, iterator, optimizer, torch.device("cpu"))
    """
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
