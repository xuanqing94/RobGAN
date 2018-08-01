import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _downsample(x):
    return F.avg_pool2d(x, 2)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None,
            ksize=3, pad=1, activation=F.relu, downsample=False, bn=False):
        super(Block, self).__init__()
        self.activation = activation
        self.bn = bn
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.Conv2d(in_channels, hidden_channels, ksize, 1, pad, bias=False)
        nn.init.xavier_uniform_(self.c1.weight, gain=math.sqrt(2.0))
        self.c2 = nn.Conv2d(hidden_channels, out_channels, ksize, 1, pad, bias=False)
        nn.init.xavier_uniform_(self.c2.weight, gain=math.sqrt(2.0))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
            nn.init.xavier_uniform_(self.c_sc.weight, gain=1.0)
        if self.bn:
            self.b1 = nn.BatchNorm2d(hidden_channels)
            nn.init.constant_(self.b1.weight, 1.0)
            self.b2 = nn.BatchNorm2d(out_channels)
            nn.init.constant_(self.b2.weight, 1.0)
            if self.learnable_sc:
                self.b_sc = nn.BatchNorm2d(out_channels)
                nn.init.constant_(self.b_sc.weight, 1.0)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.b1(self.c1(h)) if self.bn else self.c1(h)
        h = self.activation(h)
        h = self.b2(self.c2(h)) if self.bn else self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.b_sc(self.c_sc(x)) if self.bn else self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class OptimizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=F.relu, bn=False):
        super(OptimizedBlock, self).__init__()
        self.activation = activation
        self.bn = bn
        self.c1 = nn.Conv2d(in_channels, out_channels, ksize, 1, pad, bias=False)
        nn.init.xavier_uniform_(self.c1.weight, gain=math.sqrt(2.0))
        self.c2 = nn.Conv2d(out_channels, out_channels, ksize, 1, pad, bias=False)
        nn.init.xavier_uniform_(self.c2.weight, gain=math.sqrt(2.0))
        self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        nn.init.xavier_uniform_(self.c_sc.weight, gain=1.0)
        if self.bn:
            self.b1 = nn.BatchNorm2d(out_channels)
            nn.init.constant_(self.b1.weight, 1.0)
            self.b2 = nn.BatchNorm2d(out_channels)
            nn.init.constant_(self.b2.weight, 1.0)
            self.b_sc = nn.BatchNorm2d(out_channels)
            nn.init.constant_(self.b_sc.weight, 1.0)

    def residual(self, x):
        h = x
        h = self.b1(self.c1(h)) if self.bn else self.c1(h)
        h = self.activation(h)
        h = self.b2(self.c2(h)) if self.bn else self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.b_sc(self.c_sc(_downsample(x))) if self.bn else self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

