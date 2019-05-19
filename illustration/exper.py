#!/usr/bin/env python
from sklearn import datasets
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import model as m
import tcav
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

iris = datasets.load_iris()
data = m.Iris(iris.data, iris.target)

model = m.MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
iterator = DataLoader(data, 32)

for i in range(100):
    model, loss = m.train(
        model,
        iterator,
        optimizer,
        device
    )
    print(loss)


# test jacobian function
h, p = model(data[0][0])
J = tcav.mlp_jacobian(p, h)

pd.DataFrame(h.detach().numpy()).to_csv("h.csv", index=False)
pd.DataFrame(p.detach().numpy()).to_csv("p_hat.csv", index=False)
pd.DataFrame(np.hstack([iris.target.reshape([len(data), 1]), iris.data])).to_csv("iris.csv", index=False)
