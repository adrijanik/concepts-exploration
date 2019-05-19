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
iris.data = (iris.data - iris.data.mean(axis=0)) / iris.data.std(axis=0)

data = m.Iris(iris.data, iris.target)

model = MLP(n_H=100)
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

pd.DataFrame(h.detach().numpy()).to_csv("h.csv", index=False)
pd.DataFrame(p.detach().numpy()).to_csv("p_hat.csv", index=False)
pd.DataFrame(np.hstack([iris.target.reshape([len(data), 1]), iris.data])).to_csv("iris.csv", index=False)

pd.DataFrame(model.xh.weight.detach().numpy()).to_csv("w1.csv", index=False)
pd.DataFrame(model.xh.bias.detach().numpy()).to_csv("b1.csv", index=False)


# evaluate the predictions on a large grid
eval_pts = []
grid = np.linspace(-3, 3, 100)
for i in range(len(grid)):
    for j in range(len(grid)):
        x_cur = torch.Tensor([0, grid[i], grid[j], 0])
        _, p = model(x_cur)
        p = p.detach().numpy()
        print(np.argmax(p))
        eval_pts.append({
            "x0": grid[i],
            "x1": grid[j],
            "p0": p[0],
            "p1": p[1],
            "p2": p[2]
        })

pd.DataFrame(eval_pts).to_csv("eval_pts.csv", index=False)

J = all_jacobians(model, data)
scores = []

for j in range(100):
    v = np.zeros(100)
    v[j] = 1
    scores_cur = pd.DataFrame(concept_scores(J, v))
    scores_cur["sample"] = np.arange(len(scores_cur))
    scores_cur["vej"] = j
    scores.append(scores_cur)


scores = pd.concat(scores)
scores.to_csv("scores.csv", index=False)
