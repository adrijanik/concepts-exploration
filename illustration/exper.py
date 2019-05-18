#!/usr/bin/env python
from sklearn import datasets
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

iris = datasets.load_iris()
data = Iris(iris.data, iris.target)

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
iterator = DataLoader(data, 32)

for i in range(100):
    model, loss = train(
        model,
        iterator,
        optimizer,
        device
    )
    print(loss)


# test jacobian function
h, p = model(data[0][0])
J = mlp_jacobian(p, h)

pd.DataFrame(out[0].detach().numpy()).to_csv("h.csv", index=False)
pd.DataFrame(out[1].detach().numpy()).to_csv("p_hat.csv", index=False)
pd.DataFrame(np.hstack([iris.target.reshape([n, 1]), iris.data])).to_csv("iris.csv", index=False)
