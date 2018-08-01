import numpy
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class CondBatchNorm2d(nn.Module):
    def __init__(self, size, decay=0.9, eps=2.0e-5):
        super(CondBatchNorm2d, self).__init__()
        self.size = size
        self.eps = eps
        self.decay = decay
        self.register_buffer('avg_mean', torch.zeros(size))
        self.register_buffer('avg_var', torch.ones(size))
        self.register_buffer('gamma_', torch.ones(size))
        self.register_buffer('beta_', torch.zeros(size))

    def forward(self, x, gamma, beta):
        # Intentionally set self.weight == ones and self.bias == zeros
        # because we only want to normalize the input.
        feature = F.batch_norm(x, self.avg_mean, self.avg_var, Variable(self.gamma_), Variable(self.beta_), self.training, self.decay, self.eps)
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1)
        return gamma * feature + beta

class CatCondBatchNorm2d(CondBatchNorm2d):
    def __init__(self, size, n_cat, decay=0.9, eps=2.0e-5, initGamma=1.0, initBeta=0):
        super(CatCondBatchNorm2d, self).__init__(size, decay=decay, eps=eps)
        self.gammas = nn.Embedding(n_cat, size)
        nn.init.constant_(self.gammas.weight, initGamma)
        self.betas = nn.Embedding(n_cat, size)
        nn.init.constant_(self.betas.weight, initBeta)

    def forward(self, x, c):
        gamma_c = self.gammas(c)
        beta_c = self.betas(c)
        return super(CatCondBatchNorm2d, self).forward(x, gamma_c, beta_c)
