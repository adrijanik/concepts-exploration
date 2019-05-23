#!/usr/bin/env python
import torch


def mlp_jacobian(model, logits, h, y):
    """
    Returns Jacobian in 1-Layer MLP

    The k-j^th entry here is dp_k / dh_j for the (pre-sigmoid) activation for
    the k^th class with respect to the j^th coordinate of the hidden layer.
    Assumes that we have already defined logits as a function of h, at some
    point.

    :param logits: A length k Tensor of pre-activation logits.
    :param h: A tensor of MLP hidden units, from which the logits were computed
      as W^2h + b^2.

    Example
    -------
    >>> iris = datasets.load_iris()
    >>> data = Iris(iris.data, iris.target)
    >>> model = MLP()
    >>> h, p = model(data[0][0])
    >>> J = mlp_jacobian(p, h)
    """
    K = len(logits)
    jacobian = torch.zeros((K, len(h)))
    # https://discuss.pytorch.org/t/how-to-compute-jacobian-matrix-in-pytorch/14968
    for k in range(K):
        jacobian[k, :] = torch.autograd.grad(logits[k], h, retain_graph=True)[0]

    return jacobian.detach()


def all_jacobians(model, data):
    n = len(data)
    jacobians = []
    for i in range(n):
        h, pre_p = model(data[i][0])
        p = torch.softmax(pre_p, 0)
        jacobians.append(mlp_jacobian(model, p, h, data[i][1]))

    return jacobians


def concept_scores(jacobians, v):
    S = []
    for i in range(len(jacobians)):
        S.append(np.dot(jacobians[i].numpy(), v))
    return np.vstack(S)


def concepts_scores(jacobians, vs):
    scores = []
    for j, v in enumerate(vs):
        scores_cur = pd.DataFrame(concept_scores(jacobians, v))
        scores_cur["sample"] = np.arange(len(scores_cur))
        scores_cur["index"] = j
        scores.append(scores_cur)
    return pd.concat(scores)


def combine_data(data, model):
    n = len(data)
    elems = []
    for i in range(n):
        h, p = model(data[i][0])
        cur_elem = np.hstack((
            data[i][0].detach().numpy(),
            data[i][1].detach().numpy(),
            p.detach().numpy(),
            h.detach().numpy()
        )).reshape(1, -1)
        elems.append(cur_elem)

    col_names = ["X1", "X2", "y", "p0", "p1", "p2"] + ["h" + str(k) for k in range(len(h))]
    return pd.DataFrame(np.vstack(elems), columns=col_names)
