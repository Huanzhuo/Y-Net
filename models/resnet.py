import math
import torch
import numpy as np
import torch.nn as nn
import torchvision.models

class Conv(nn.Module):
    # Basic Conv block: conv + bn + ac_func
    def __init__(self, ch_in, ch_out, k, s=1, p=0, d=1, ac_func='Relu'):
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(ch_in, ch_out, k, s, padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm1d(ch_out)
        if ac_func == 'PRelu':
            self.ac_func = nn.PReLU()
        elif ac_func == 'Mish':
            self.ac_func = Mish()
        elif ac_func == 'LeakyRelu':
            self.ac_func = nn.LeakyReLU()
        elif ac_func == 'Relu':
            self.ac_func = nn.ReLU()
        else:
            self.ac_func = nn.Identity()

    def forward(self, x):
        return self.ac_func(self.bn(self.conv(x)))


class Conv_1(nn.Module):
    # Basic Conv block: conv + bn
    def __init__(self, ch_in, ch_out, k, s=1, p=0, d=1):
        super(Conv_1, self).__init__()
        self.conv = nn.Conv1d(ch_in, ch_out, k, s, padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm1d(ch_out)

    def forward(self, x):
        return self.bn(self.conv(x))


class Conv1d(nn.Module):
    # Basic Conv block: conv + bn + ac_func
    def __init__(self, ch_in, ch_out, k, s=1, p=0, d=1, ac_func='Relu'):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(ch_in, ch_out, k, s, padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm1d(ch_out)
        if ac_func == 'PRelu':
            self.ac_func = nn.PReLU()
        elif ac_func == 'Mish':
            self.ac_func = Mish()
        elif ac_func == 'LeakyRelu':
            self.ac_func = nn.LeakyReLU()
        elif ac_func == 'Relu':
            self.ac_func = nn.ReLU()
        else:
            self.ac_func = nn.Identity()

    def forward(self, x):
        return self.ac_func(self.bn(self.conv(x)))


class Conv_1d_1(nn.Module):
    # Basic Conv block: conv + bn
    def __init__(self, ch_in, ch_out, k, s=1, p=0, d=1):
        super(Conv_1d_1, self).__init__()
        self.conv = nn.Conv1d(ch_in, ch_out, k, s, padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm1d(ch_out)

    def forward(self, x):
        return self.bn(self.conv(x))


class FrequencyAttention(nn.Module):
    def __init__(self, chs_in, chs_out, size):
        super(FrequencyAttention, self).__init__()
        expansion = 16
        chs_mid = chs_in // expansion
        f = size
        target_size = (f, 1)
        self.avg = nn.AdaptiveAvgPool2d(target_size)
        self.se = nn.Sequential(Conv1d(chs_in, chs_mid, 1),
                                Conv1d(chs_mid, chs_mid, 3, p=1),
                                Conv_1d_1(chs_mid, chs_out, 1),
                                nn.Sigmoid())

    def forward(self, x):
        batch_size, chs, f, t = x.shape
        se = self.avg(x).view(batch_size, chs, f)
        se = self.se(se).view(batch_size, chs, f, 1)
        return x * se.expand_as(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, ac_func='Relu'):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv(inplanes, planes, k=1, s=stride)
        self.conv2 = Conv(planes, planes, k=3, p=1, s=1)
        self.conv3 = Conv_1(planes, planes * 4, k=1)

        self.downsample = downsample
        self.stride = stride
        if ac_func == 'PRelu':
            self.ac_func = nn.PReLU()
        elif ac_func == 'Mish':
            self.ac_func = Mish()
        elif ac_func == 'LeakyRelu':
            self.ac_func = nn.LeakyReLU()
        elif ac_func == 'Relu':
            self.ac_func = nn.ReLU()
        else:
            self.ac_func = nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.ac_func(out)

        return out
