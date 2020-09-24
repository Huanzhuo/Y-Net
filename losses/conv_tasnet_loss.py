import numpy as np
import torch.nn as nn
from itertools import permutations
from torch.autograd import Variable
import scipy, time, numpy
import itertools
import torch.nn.functional as F
import torch


def sdr_object(estimation, origin, masks):
    """
    batch-wise SDR caculation for one audio file.
    estimation: (batch, chs, nsample)
    origin: (batch, chs, nsample)
    mask: optional, (batch, chs), binary  True if audio is silence
    """
    estimation = estimation.unsqueeze(2) # (batch, chs, 1, nsample)
    origin = origin.unsqueeze(2)
    origin_power = torch.pow(origin, 2).sum(dim=-1, keepdim=True) + 1e-8  # shape: (B, 4, 1, 1)
    scale = torch.sum(origin * estimation, dim=-1, keepdim=True) / origin_power  # shape: (B, 4, 1, 1)

    est_true = scale * origin
    est_res = estimation - est_true

    true_power = torch.pow(est_true, 2).sum(dim=-1).clamp(min=1e-8)
    res_power = torch.pow(est_res, 2).sum(dim=-1).clamp(min=1e-8)

    sdr = 10 * (torch.log10(true_power) - torch.log10(res_power))

    if masks is not None:
        masks = masks.unsqueeze(2)
        sdr = (sdr * masks).sum(dim=(0, -1)) / masks.sum(dim=(0, -1)).clamp(min=1e-8)  # shape: (4)
        # masks = masks.unsqueeze(2).unsqueeze(3)
        # sdr = (sdr * masks).sum(dim=(0, -1)) / masks.sum(dim=(0, -1)).clamp(min=1e-8)  # shape: (4)
    else:
        sdr = sdr.mean(dim=(0, -1))  # shape: (4)

    return sdr


def cal_kl_loss(estimation, origin, masks):
    masks = masks.unsqueeze(-1)
    loss = F.kl_div(estimation, origin, reduction='none')
    loss = (loss * masks).sum(dim=(0, -1)) / masks.sum(dim=(0, -1)).clamp(min=1e-8)
    return loss


def calculate_loss(estimation, origin, mix, mask=None, ratio=0.5, reconst=False):
    sdr = sdr_object(estimation, origin, mask)
    loss = -sdr.sum()

    if reconst:
        estimated_mix = estimation.sum(1)
        if mask is not None:
            reconstruction_sdr = sdr_object(estimated_mix.unsqueeze(1), mix.unsqueeze(1), mask).mean()
        else:
            reconstruction_sdr = sdr_object(estimated_mix.unsqueeze(1), mix.unsqueeze(1), mask).mean()
        loss += -ratio * reconstruction_sdr
    # loss = loss * 0.5 + l1_loss
    return loss