#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from torch.autograd import grad, Variable
from .linf_sgd import Linf_SGD

# performs Linf-constraint PGD attack w/o noise
# @epsilon: radius of Linf-norm ball
def attack_Linf_PGD(input_v, ones, label_v, dis, Ld, steps, epsilon):
    dis.eval()
    adverse_v = input_v.data.clone()
    adverse_v = Variable(adverse_v, requires_grad=True)
    optimizer = Linf_SGD([adverse_v], lr=0.0078)
    for _ in range(steps):
        optimizer.zero_grad()
        dis.zero_grad()
        d_bin, d_multi = dis(adverse_v)
        loss = -Ld(d_bin, ones, d_multi, label_v, lam=0.5)
        loss.backward()
        #print(loss.data[0])
        optimizer.step()
        diff = adverse_v.data - input_v.data
        diff.clamp_(-epsilon, epsilon)
        adverse_v.data.copy_((diff + input_v.data).clamp_(-1, 1))
    dis.train()
    dis.zero_grad()
    return adverse_v

# performs Linf-constraint PGD attack w/o noise
# @epsilon: radius of Linf-norm ball
def attack_Linf_PGD_bin(input_v, ones, dis, Ld, steps, epsilon):
    dis.eval()
    adverse_v = input_v.data.clone()
    adverse_v = Variable(adverse_v, requires_grad=True)
    optimizer = Linf_SGD([adverse_v], lr=0.0078)
    for _ in range(steps):
        optimizer.zero_grad()
        dis.zero_grad()
        d_bin = dis(adverse_v)
        loss = -Ld(d_bin, ones)
        loss.backward()
        #print(loss.data[0])
        optimizer.step()
        diff = adverse_v.data - input_v.data
        diff.clamp_(-epsilon, epsilon)
        adverse_v.data.copy_((diff + input_v.data).clamp_(-1, 1))
    dis.train()
    dis.zero_grad()
    return adverse_v

# performs FGSM attack, and it is differentiable
# @input_v: make sure requires_grad = True
def attack_FGSM(input_v, ones, label_v, dis, Lg):
    dis.eval()
    d_bin, d_multi = dis(input_v)
    loss = -Lg(d_bin, ones, d_multi, label_v, lam=0.5)
    g = grad(loss, [input_v], create_graph=True)[0]
    return input_v - 0.005 * torch.sign(g)


# performs Linf-constraint PGD attack w/o noise
# @epsilon: radius of Linf-norm ball
def attack_label_Linf_PGD(input_v, label_v, dis, steps, epsilon):
    dis.eval()
    adverse_v = input_v.data.clone()
    adverse_v = Variable(adverse_v, requires_grad=True)
    optimizer = Linf_SGD([adverse_v], lr=epsilon / 5)
    for _ in range(steps):
        optimizer.zero_grad()
        dis.zero_grad()
        _, d_multi = dis(adverse_v)
        loss = -F.cross_entropy(d_multi, label_v)
        loss.backward()
        #print(loss.data[0])
        optimizer.step()
        diff = adverse_v.data - input_v.data
        diff.clamp_(-epsilon, epsilon)
        adverse_v.data.copy_((diff + input_v.data).clamp_(-1, 1))
    dis.zero_grad()
    return adverse_v


