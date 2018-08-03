import torch
import torch.nn as nn
import torch.nn.functional as F
from .resblocks import Block, OptimizedBlock

class ResNetAC(nn.Module):
    def __init__(self, ch=64, n_classes=0, activation=F.relu, bn=False):
        super(ResNetAC, self).__init__()
        self.activation = activation
        self.block1 = OptimizedBlock(3, ch, bn=bn)
        self.block2 = Block(ch, ch * 2, activation=activation, downsample=True, bn=bn)
        self.block3 = Block(ch * 2, ch * 4, activation=activation, downsample=True, bn=bn)
        self.block4 = Block(ch * 4, ch * 8, activation=activation, downsample=True, bn=bn)
        self.block5 = Block(ch * 8, ch * 16, activation=activation, downsample=True, bn=bn)
        self.block6 = Block(ch * 16, ch * 16, activation=activation, downsample=False, bn=bn)
        self.l7 = nn.Linear(ch * 16, 1)
        nn.init.xavier_uniform_(self.l7.weight, gain=1.0)
        if n_classes > 0:
            self.l_y = nn.Linear(ch * 16, n_classes)
            nn.init.xavier_uniform_(self.l_y.weight, gain=1.0)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)
        # global sum pooling (h, w)
        #TODO try to use global avg pooling instead
        h = h.view(h.size(0), h.size(1), -1)
        h = torch.sum(h, 2)
        output = self.l7(h)
        w_y = self.l_y(h)
        return output.view(-1), w_y
