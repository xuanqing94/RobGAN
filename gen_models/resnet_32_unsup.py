import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class UpResBlock(nn.Module):
    def __init__(self, ch):
        super(UpResBlock, self).__init__()
        self.c0 = nn.Conv2d(ch, ch, 3, 1, 1)
        nn.init.normal_(self.c0.weight, 0.02)
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1)
        nn.init.normal_(self.c1.weight, 0.02)
        self.cs = nn.Conv2d(ch, ch, 3, 1, 1)
        nn.init.normal_(self.cs.weight, 0.02)
        self.bn0 = nn.BatchNorm2d(ch)
        nn.init.constant_(self.bn0.weight, 1.0)
        nn.init.constant_(self.bn0.bias, 0.0)
        self.bn1 = nn.BatchNorm2d(ch)
        nn.init.constant_(self.bn0.weight, 1.0)
        nn.init.constant_(self.bn0.bias, 0.0)

    @classmethod
    def upsample(cls, x):
        h, w = x.shape[2:]
        return F.upsample(x, size=(h * 2, w * 2))

    def forward(self, x):
        h = self.c0(upsample(F.relu(self.bn0(x))))
        h = self.c1(F.relu(self.bn1(h)))
        hs = self.cs(upsample(x))
        return h + hs

class ResNetGenerator(nn.Module):
    def __init__(self, ch=64, dim_z=128, bottom_width=4):
        super(ResNetGenerator, self).__init__()
        self.bottom_width = bottom_width
        self.dim_z = dim_z
        self.ch = ch
        self.l0 = nn.Linear(dim_z, (bottom_width ** 2) * ch * 4)
        nn.init.normal_(self.l0.weight, math.sqrt(1.0 / dim_z))
        self.r0 = UpResBlock(ch * 4)
        self.r1 = UpResBlock(ch * 4)
        self.r2 = UpResBlock(ch * 4)
        self.bn2 = nn.BatchNorm2d(ch * 4)
        nn.init.constant_(self.bn2.weight, 1.0)
        nn.init.constant_(self.bn2.bias, 0.0)
        self.c3 = nn.Conv2d(ch * 4, 3, 3, 1, 1)
        nn.init.normal_(self.c3.weight, 0.02)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = h.view(-1, self.ch * 4, self.bottom_width, self.bottom_width)
        h = self.r0(h)
        h = self.r1(h)
        h = self.r2(h)
        h = self.bn2(F.relu(h))
        h = self.c3(h)
        h = F.tanh(h)
        return h
