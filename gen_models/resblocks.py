import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.cat_cond_bn import CatCondBatchNorm2d


def _upsample(x):
    h, w = x.shape[2:]
    return F.upsample(x, size=(h * 2, w * 2))

def upsample_conv(x, conv):
    return conv(_upsample(x))

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, \
            pad=1, activation=F.relu, upsample=False, n_classes=0):
        super(Block, self).__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        nn.init.xavier_uniform_(self.c1.weight, math.sqrt(2.0))
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)
        nn.init.xavier_uniform_(self.c2.weight, math.sqrt(2.0))
        if n_classes > 0:
            self.b1 = CatCondBatchNorm2d(in_channels, n_cat=n_classes)
            self.b2 = CatCondBatchNorm2d(hidden_channels, n_cat=n_classes)
        else:
            self.b1 = nn.BatchNorm2d(in_channels)
            nn.init.constant_(self.b1.weight, 1.0)
            self.b2 = nn.BatchNorm2d(hidden_channels)
            nn.init.constant_(self.b2.weight, 1.0)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            nn.init.xavier_uniform_(self.c_sc.weight, 1.0)

    def residual(self, x, y=None):
        h = x
        h = self.b1(h, y) if y is not None else self.b1(h)
        h = self.activation(h)
        h = upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h, y) if y is not None else self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x, y=None):
        f1 = self.residual(x, y)
        f2 = self.shortcut(x)
        return f1 + f2







