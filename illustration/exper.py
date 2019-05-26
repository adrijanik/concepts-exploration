#!/usr/bin/env python
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
import model as m
import numpy as np
import pandas as pd
import tcav
import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(100)

# generate some data
n = 2000
x = np.random.uniform(-1, 1, size=(n, 2))
y = np.zeros(n)
circle_ix = x[:, 0] ** 2 + x[:, 1] ** 2 <= .25 ** 2
y[circle_ix] = 1
y[np.logical_not(circle_ix) * x[:, 1] > 0] = 2
data = m.DF(x, y.astype(np.int64))

# fit a simple model
H = 20
model = m.MLP(n_H=H, p=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
iterator = DataLoader(data, 32)

for i in range(80):
    model, loss = m.train(model, iterator, optimizer, device)
    if i % 20 == 0:
        print(loss)

# get the directional derivatives, evaluated along different standard basis
J = tcav.all_jacobians(model, data)
vs = [np.eye(1, H, k).squeeze() for k in range(H)]
scores = concepts_scores(J, vs)
scores.to_csv("scores.csv", index=False)

# evaluate the predictions on a large grid
eval_pts = eval_grid(model)
pd.DataFrame(eval_pts).to_csv("eval_pts.csv", index=False)

# compute concept activation in top eigenvector directions
combined = combine_data(data, model)
combined.to_csv("combined.csv", index=False)

# write the parameters for reference
pd.DataFrame(model.xh.weight.detach().numpy()).to_csv("w1.csv", index=False)
pd.DataFrame(model.xh.bias.detach().numpy()).to_csv("b1.csv", index=False)
pd.DataFrame(model.hy.weight.detach().numpy()).to_csv("w2.csv", index=False)
pd.DataFrame(model.hy.bias.detach().numpy()).to_csv("b2.csv", index=False)

# directional derivatives, on random directions in H dimensional space
_, _, v = np.linalg.svd(combined.iloc[:, 6:].values)
vs = []
for j in range(500):
    coefs = np.random.normal(0, 1, H)
    coefs /= np.sqrt(sum(coefs ** 2))
    # vs.append(np.dot(v[:, :3], coefs))
    vs.append(coefs)

tcav.concepts_scores(J, vs).to_csv("scores_rand.csv", index=False)
pd.DataFrame(vs).to_csv("v_rand.csv", index=False)

kmeans = KMeans(n_clusters=50).fit(x)

# fix some direction
v = np.eye(1, H, 0).squeeze()

scores = []
for k in range(50):
    ix = kmeans.labels_ == k
    cur_data = []
    for i in np.where(ix)[0]:
        cur_data.append(data[i])

    J = tcav.all_jacobians(model, cur_data)
    cur_scores = pd.DataFrame(tcav.concept_scores(J, v))
    cur_scores["sample"] = np.where(ix)[0]
    cur_scores["k"] = k
    scores.append(cur_scores)

pd.concat(scores).to_csv("scores_cluster.csv", index=False)
pd.DataFrame(kmeans.cluster_centers_).to_csv("cluster_centroids.csv", index=False)

scores
