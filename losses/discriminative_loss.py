import numpy as np
import torch.nn as nn
from itertools import permutations
from torch.autograd import Variable
import scipy, time, numpy
import itertools
import torch.nn.functional as F
import torch


def var_loss(features, masks, theta_v=1):
    masks = masks.unsqueeze(-1)
    if len(features.shape) == 4:
        B, SPK, _, _ = features.shape
        features = features.view(B, SPK, -1)
        N = features.shape[-1]
    else:
        B, SPK, N = features.shape
    u = features.mean(dim=-1).unsqueeze(-1)
    var = torch.abs(features - u) - theta_v
    loss_var = torch.square(var.ge(0) * var)
    loss_var = (loss_var * masks).sum() / masks.sum()
    return loss_var


def dist_loss(features, masks, theta_d=1.5):
    masks = masks.unsqueeze(-1)
    u = features.mean(dim=-1).unsqueeze(-1)
    SPK = u.shape[1]
    u_ = u.repeat(1, 1, SPK)
    masks = masks.repeat(1, 1, SPK)
    dist = 2 * theta_d-torch.abs(u_.transpose(1,2) - u)
    dist = torch.square(dist.ne(2*theta_d) * dist.ge(0) * dist) * masks
    loss_dist = dist.sum() / (dist.ne(0).sum())
    return loss_dist


def reg_loss(features, masks):
    u = features.mean(dim=-1)
    loss_reg = torch.abs(u * masks).sum() / masks.sum().clamp(min=1e-8)
    return loss_reg


if __name__ == '__main__':
    a = torch.zeros([8,4])
    features = torch.randn([2, 4, 256])
    masks = torch.tensor([[1, 1, 0, 1], [1, 1, 0, 1]])
    loss_var = var_loss(features, masks).mean()
    loss_dist = dist_loss(features, masks)
    loss_reg = reg_loss(features, masks)
    print('test')




