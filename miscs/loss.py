import torch
import torch.nn.functional as F

# Classic adversarial loss
def loss_KL_d(dis_fake, dis_real):
    L1 = torch.mean(F.softplus(-dis_real))
    L2 = torch.mean(F.softplus(dis_fake))
    return L1 + L2

def loss_KL_g(dis_fake):
    return torch.mean(F.softplus(-dis_fake))


# Hinge loss
def loss_hinge_d(dis_fake, dis_real):
    L1 = torch.mean(F.relu(1 - dis_real))
    L2 = torch.mean(F.relu(1 + dis_fake))
    return L1 + L2

def loss_hinge_g(dis_fake):
    return -torch.mean(dis_fake)


# NLL loss
def loss_nll(bin_output, bin_label, multi_output, multi_label, lam=0.5):
    L1 = F.binary_cross_entropy_with_logits(bin_output, bin_label)
    L2 = F.cross_entropy(multi_output, multi_label)
    return lam * L1 + (1.0 - lam) * L2

# NLL loss with another weighting scheme
def loss_nll_v2(bin_output, bin_label, multi_output, multi_label, lam):
    L1 = F.binary_cross_entropy_with_logits(bin_output, bin_label)
    L2 = F.cross_entropy(multi_output, multi_label)
    return L1 + lam * L2

# Binary loss
def loss_bin(bin_output, bin_label):
    return F.binary_cross_entropy_with_logits(bin_output, bin_label)
