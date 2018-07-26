#!/usr/bin/env python

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as tfs
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--model_in', type=str, required=True)
parser.add_argument('--ndf', type=int, required=True)
parser.add_argument('--nclass', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--root', type=str, required=True)
parser.add_argument('--img_width', type=int, required=True)
parser.add_argument('--ngpu', type=int, required=True)
parser.add_argument('--workers', type=int, default=3)
opt = parser.parse_args()

def load_model():
    if opt.model == "resnet_64":
        from dis_models.resnet_64 import ResNetAC
        dis = ResNetAC(ch=opt.ndf, n_classes=opt.nclass)
    elif opt.model == "resnet_128":
        from dis_models.resnet_small import ResNetAC
        dis = ResNetAC(ch=opt.ndf, n_classes=opt.nclass)
    elif opt.model == "resnet_imagenet":
        from dis_models.resnet import ResNetAC
        dis = ResNetAC(ch=opt.ndf, n_classes=opt.nclass)
    else:
        raise ValueError("Unknown model name: {}".format(opt.model))
    if opt.ngpu > 0:
        dis = dis.cuda()
        dis = torch.nn.DataParallel(dis, device_ids=range(opt.ngpu))
    else:
        raise ValueError("Must run on gpus, ngpu > 0")
    dis.load_state_dict(torch.load(opt.model_in))
    return dis

def make_dataset():
    trans = tfs.Compose([
        tfs.Resize(opt.img_width),
        tfs.ToTensor(),
        tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
    data = ImageFolder(opt.root, transform=trans)
    loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    return loader

def main():
    # model
    dis = load_model()
    dis.eval()
    # data
    test_loader = make_dataset()
    correct_real = 0
    correct_label = 0
    total = 0
    with torch.no_grad():
        for x_real, y_real in test_loader:
            x_real, y_real = x_real.cuda(), y_real.cuda()
            v_x_real = Variable(x_real)
            d_bin, d_multi = dis(v_x_real)
            bin_correct = d_bin.data > 0
            _, idx = torch.max(d_multi.data, dim=1)
            label_correct = idx.eq(y_real)
            #correct_label += torch.sum(bin_correct * label_correct)
            correct_label += torch.sum(label_correct)
            #total += torch.sum(bin_correct)
            total += y_real.numel()
    print('Accuracy: {}'.format(correct_label / total))

if __name__ == "__main__":
    main()
