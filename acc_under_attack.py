#!/usr/bin/env python

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as tfs
from torchvision.datasets import ImageFolder, CIFAR10
from torch.utils.data import DataLoader
from miscs.pgd import attack_label_Linf_PGD

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--model_in', type=str, required=True)
parser.add_argument('--ndf', type=int, required=True)
parser.add_argument('--nclass', type=int, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--root', type=str, required=True)
parser.add_argument('--img_width', type=int, required=True)
parser.add_argument('--steps', type=int, required=True)
parser.add_argument('--epsilon', type=str, required=True)
parser.add_argument('--ngpu', type=int, required=True)
parser.add_argument('--workers', type=int, default=3)
opt = parser.parse_args()

opt.epsilon = [float(e) for e in opt.epsilon.split(',')]

def load_model():
    if opt.model == "resnet_32":
        from dis_models.preact_resnet import PreActResNet18
        dis = PreActResNet18()
    elif opt.model == "resnet_64":
        from dis_models.resnet_64 import ResNetAC
        dis = ResNetAC(ch=opt.ndf, n_classes=opt.nclass)
    elif opt.model == "resnet_128":
        from dis_models.resnet_small import ResNetAC
        dis = ResNetAC(ch=opt.ndf, n_classes=opt.nclass)
    elif opt.model == "resnet_imagenet":
        from dis_models.resnet import ResNetAC
        dis = ResNetAC(ch=opt.ndf, n_classes=opt.nclass)
    else:
        raise ValueError(f"Unknown model name: {opt.model}")
    if opt.ngpu > 0:
        dis = dis.cuda()
        dis = torch.nn.DataParallel(dis, device_ids=range(opt.ngpu))
    else:
        raise ValueError("Must run on gpus, ngpu > 0")
    dis.load_state_dict(torch.load(opt.model_in))
    dis.eval()
    return dis

def make_dataset():
    if opt.dataset in ("imagenet", "dog_and_cat_64", "dog_and_cat_128"):
        trans = tfs.Compose([
            tfs.Resize(opt.img_width),
            tfs.ToTensor(),
            tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
        data = ImageFolder(opt.root, transform=trans)
        loader = DataLoader(data, batch_size=100, shuffle=False, num_workers=opt.workers)
    elif opt.dataset == "cifar10":
        trans = tfs.Compose([
            tfs.Resize(opt.img_width),
            tfs.ToTensor(),
            tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
        data = CIFAR10(root=opt.root, train=True, download=False, transform=trans)
        loader = DataLoader(data, batch_size=100, shuffle=True, num_workers=opt.workers)
    else:
        raise ValueError(f"Unknown dataset: {opt.dataset}")
    return loader


def main(epsilon):
    # model
    dis = load_model()
    dis.eval()
    # data
    loader = make_dataset()
    correct_real = 0
    correct_label = 0
    total = 0

    for i, (x_real, y_real) in enumerate(loader):
        if i == 100:
            break
        x_real, y_real = x_real.cuda(), y_real.cuda()
        v_y_real, v_x_real = Variable(y_real), Variable(x_real)
        adv_input = attack_label_Linf_PGD(v_x_real, v_y_real, dis, opt.steps, epsilon)
        with torch.no_grad():
            _, d_multi = dis(adv_input)
        _, idx = torch.max(d_multi.data, dim=1)
        label_correct = idx.eq(y_real)
        correct_label += torch.sum(label_correct)
        total += y_real.numel()
    print(f'{epsilon}, {correct_label/total}')

if __name__ == "__main__":
    print('#c, accuracy')
    for epsilon in opt.epsilon:
        main(epsilon)
