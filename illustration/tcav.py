#!/usr/bin/env python
import torch


def mlp_jacobian(model, logits, h):
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
    x = h
    x.detach()
    logit = model.hy(x)
    jacobian = torch.zeros((K, len(x)))
    for k in range(len(logit)):
        ek = torch.zeros(K)
        ek[k] = 1
        jacobian[k, ] = torch.matmul((ek - logit), model.hy.weight).detach()

    # jacobian2 = torch.zeros((K, len(h)))
    # # https://discuss.pytorch.org/t/how-to-compute-jacobian-matrix-in-pytorch/14968
    # for k in range(K):
    #     jacobian2[k, :] = torch.autograd.grad(logits[k], h, logits[k], retain_graph=True)[0]
    return jacobian


def all_jacobians(model, data):
    n = len(data)
    jacobians = []
    for i in range(n):
        h = model.xh(data[i][0])
        p = model.hy(h)
        jacobians.append(mlp_jacobian(model, p, h))

    return jacobians


def collect_stats(model, data):
    n = len(data)
    stats = {"h": [], "p": []}
    for i in range(n):
        h, p = model(data[i][0])
        stats["h"].append(h)
        stats["p"].append(p)

    return stats


def concept_scores(jacobians, v):
    S = []
    for i in range(len(jacobians)):
        S.append(np.dot(jacobians[i].numpy(), v))

    return np.vstack(S)


from torch.autograd.gradcheck import zero_gradients

def compute_jacobian(inputs, output):
	"""
	:param inputs: Batch X Size (e.g. Depth X Width X Height)
	:param output: Batch X Classes
	:return: jacobian: Batch X Classes X Size
	"""
	num_classes = output.size()[0]

	jacobian = torch.zeros(num_classes, *inputs.size())
	grad_output = torch.zeros(*output.size())
	if inputs.is_cuda:
		grad_output = grad_output.cuda()
		jacobian = jacobian.cuda()

	for i in range(num_classes):
		zero_gradients(inputs)
		grad_output.zero_()
		grad_output[i] = 1
		output.backward(grad_output, retain_graph=True)
		jacobian[i] = inputs.grad.data

	return torch.transpose(jacobian, dim0=0, dim1=1)

