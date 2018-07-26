import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, ch=64, dim_z=128, bottom_width=4):
        super(Generator, self).__init__()
        self.bottom_width = bottom_width
        self.ch = ch
        self.l0 = nn.Linear(dim_z, (bottom_width ** 2) * ch * 8)
        self.bn0 = nn.BatchNorm1d(bottom_width * bottom_width * ch * 8)

        self.dc1 = nn.ConvTranspose2d(ch * 8, ch * 4, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ch * 4)

        self.dc2 = nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ch * 2)

        self.dc3 = nn.ConvTranspose2d(ch * 2, ch,     4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ch)

        self.dc4 = nn.ConvTranspose2d(ch,     3,      3, 1, 1)

    def forward(self, z):
        out = F.relu(self.bn0(self.l0(z)))
        out = out.view(-1, self.ch * 8, self.bottom_width, self.bottom_width)
        out = F.relu(self.bn1(self.dc1(out)))
        out = F.relu(self.bn2(self.dc2(out)))
        out = F.relu(self.bn3(self.dc3(out)))
        out = F.tanh(self.dc4(out))
        return out
