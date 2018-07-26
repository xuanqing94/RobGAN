import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, ch=64, bn=False):
        super(Discriminator, self).__init__()
        self.bn = bn
        self.ch = ch
        self.c0_0 = nn.Conv2d(3     , ch    , 3, 1, 1)
        self.c0_1 = nn.Conv2d(ch    , ch * 2, 4, 2, 1)
        self.c1_0 = nn.Conv2d(ch * 2, ch * 2, 3, 1, 1)
        self.c1_1 = nn.Conv2d(ch * 2, ch * 4, 4, 2, 1)
        self.c2_0 = nn.Conv2d(ch * 4, ch * 4, 3, 1, 1)
        self.c2_1 = nn.Conv2d(ch * 4, ch * 8, 4, 2, 1)
        self.c3_0 = nn.Conv2d(ch * 8, ch * 8, 3, 1, 1)
        self.l4 = nn.Linear(4 * 4 * ch * 8, 1)

        if self.bn:
            self.bn0_1 = nn.BatchNorm2d(ch * 2)
            self.bn1_0 = nn.BatchNorm2d(ch * 2)
            self.bn1_1 = nn.BatchNorm2d(ch * 4)
            self.bn2_0 = nn.BatchNorm2d(ch * 4)
            self.bn2_1 = nn.BatchNorm2d(ch * 8)
            self.bn3_0 = nn.BatchNorm2d(ch * 8)

    def forward(self, x):
        if self.bn:
            out = self.c0_0(x)
            out = F.leaky_relu(out, negative_slope=0.2)
            out = self.bn0_1(self.c0_1(out))
            out = F.leaky_relu(out, negative_slope=0.2)
            out = self.bn1_0(self.c1_0(out))
            out = F.leaky_relu(out, negative_slope=0.2)
            out = self.bn1_1(self.c1_1(out))
            out = F.leaky_relu(out, negative_slope=0.2)
            out = self.bn2_0(self.c2_0(out))
            out = F.leaky_relu(out, negative_slope=0.2)
            out = self.bn2_1(self.c2_1(out))
            out = F.leaky_relu(out, negative_slope=0.2)
            out = self.bn3_0(self.c3_0(out))
            out = F.leaky_relu(out, negative_slope=0.2)
            out = out.view(-1, 4 * 4 * self.ch * 8)
        else:
            out = self.c0_0(x)
            out = F.leaky_relu(out, negative_slope=0.2)
            out = self.c0_1(out)
            out = F.leaky_relu(out, negative_slope=0.2)
            out = self.c1_0(out)
            out = F.leaky_relu(out, negative_slope=0.2)
            out = self.c1_1(out)
            out = F.leaky_relu(out, negative_slope=0.2)
            out = self.c2_0(out)
            out = F.leaky_relu(out, negative_slope=0.2)
            out = self.c2_1(out)
            out = F.leaky_relu(out, negative_slope=0.2)
            out = self.c3_0(out)
            out = F.leaky_relu(out, negative_slope=0.2)
            out = out.view(-1, 4 * 4 * self.ch * 8)
        out = self.l4(out)
        return out.view(-1)
