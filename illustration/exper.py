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
# pd.DataFrame(np.hstack([iris.target.reshape([len(data), 1]), iris.data])).to_csv("iris.csv", index=False)
H = 500

model = MLP(n_H=H)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
iterator = DataLoader(data, 32)

for i in range(1000):
    model, loss = train(
        model,
        iterator,
        optimizer,
        device
    )
    if i % 100 == 0:
        print(loss)


# evaluate the predictions on a large grid
eval_pts = []
grid = np.linspace(-3, 3, 100)
for i in range(len(grid)):
    for j in range(len(grid)):
        x_cur = torch.Tensor([0, grid[i], grid[j], 0])
        _, p = model(x_cur)
        p = p.detach().numpy()
        eval_pts.append({
            "x0": grid[i],
            "x1": grid[j],
            "p0": p[0],
            "p1": p[1],
        })

pd.DataFrame(eval_pts).to_csv("eval_pts.csv", index=False)

# get the directional derivatives, evaluated along different standard basis
stats = collect_stats(model, data)
J = all_jacobians(model, data)
scores = []

for j in range(H):
    v = np.zeros(H)
    v[j] = 1
    scores_cur = pd.DataFrame(concept_scores(J, v))
    scores_cur["sample"] = np.arange(len(scores_cur))
    scores_cur["vej"] = j
    scores.append(scores_cur)

scores = pd.concat(scores)
scores.to_csv("scores.csv", index=False)

# compute concept activation in top eigenvector directions
stats["h"] = pd.DataFrame(stats["h"])
stats["p"] = pd.DataFrame(stats["p"])

pd.DataFrame(model.xh.weight.detach().numpy()).to_csv("w1.csv", index=False)
h = [z.detach().numpy() for z in stats["h"]]
h = pd.DataFrame(np.vstack(h)).to_csv("h.csv", index=False)
p = pd.concat([pd.DataFrame(z.detach().numpy()) for z in stats["p"]])

# directional derivatives, on random directions in H dimensional space
scores = []
for j in range(10000):
    v = np.random.randn(H)
    v /= np.sqrt(sum(v ** 2))
    scores_cur = pd.DataFrame(concept_scores(J, v))
    scores_cur["sample"] = np.arange(len(scores_cur))
    scores_cur["draw"] = j
    scores.append(scores_cur)

scores = pd.concat(scores)
scores.to_csv("scores_rand.csv", index=False)
