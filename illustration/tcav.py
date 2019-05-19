#!/usr/bin/env python
import torch


def mlp_jacobian(logits, h):
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
        jacobian[k, :] = torch.autograd.grad(logits[k], h, logits[k], retain_graph=True)[0]

    return jacobian

def all_jacobians(model, data):
    n = len(data)
    jacobians = []
    for i in range(n):
        h, p = model(data[i][0])
        jacobians.append(mlp_jacobian(p, h))

    return jacobians

def concept_scores(jacobians, v):
    S = []
    for i in range(len(jacobians)):
        S.append(np.dot(jacobians[i].numpy(), v))

    return np.vstack(S)
