#!/usr/bin/env python

import os, sys, time
import shutil
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.datasets import ImageFolder, CIFAR10
import torchvision.transforms as tfs
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from miscs.pgd import attack_Linf_PGD, attack_FGSM
from miscs.loss import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--nz', type=int, required=True)
parser.add_argument('--ngf', type=int, required=True)
parser.add_argument('--ndf', type=int, required=True)
parser.add_argument('--nclass', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--start_width', type=int, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--root', type=str, required=True)
parser.add_argument('--img_width', type=int, required=True)
parser.add_argument('--iter_d', type=int, default=5)
parser.add_argument('--out_f', type=str, required=True)
parser.add_argument('--ngpu', type=int, required=True)
parser.add_argument('--workers', type=int, default=3)
parser.add_argument('--starting_epoch', type=int, default=0)
parser.add_argument('--max_epoch', type=int, required=True)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--adv_steps', type=int, required=True)
parser.add_argument('--epsilon', type=float, required=True)
opt = parser.parse_args()

def load_models():
    if opt.model == "resnet_32":
        from gen_models.resnet_32 import ResNetGenerator
        from dis_models.resnet_32 import ResNetAC
        gen = ResNetGenerator(ch=opt.ngf, dim_z=opt.nz, bottom_width=opt.start_width, n_classes=opt.nclass)
        dis = ResNetAC(ch=opt.ndf, n_classes=opt.nclass, bn=True)
    elif opt.model == "resnet_64":
        from gen_models.resnet_64 import ResNetGenerator
        from dis_models.resnet_64 import ResNetAC
        gen = ResNetGenerator(ch=opt.ngf, dim_z=opt.nz, bottom_width=opt.start_width, n_classes=opt.nclass)
        dis = ResNetAC(ch=opt.ndf, n_classes=opt.nclass)
    elif opt.model == "resnet_128":
        from gen_models.resnet_small import ResNetGenerator
        from dis_models.resnet_small import ResNetAC
        gen = ResNetGenerator(ch=opt.ngf, dim_z=opt.nz, bottom_width=opt.start_width, n_classes=opt.nclass)
        dis = ResNetAC(ch=opt.ndf, n_classes=opt.nclass)
    elif opt.model == "resnet_imagenet":
        from gen_models.resnet import ResNetGenerator
        from dis_models.resnet import ResNetAC
        gen = ResNetGenerator(ch=opt.ngf, dim_z=opt.nz, bottom_width=opt.start_width, n_classes=opt.nclass)
        dis = ResNetAC(ch=opt.ndf, n_classes=opt.nclass)
    else:
        raise ValueError("Unknown model name: {}".format(opt.model))
    if opt.ngpu > 0:
        gen, dis = gen.cuda(), dis.cuda()
        gen, dis = torch.nn.DataParallel(gen, device_ids=range(opt.ngpu)), \
                torch.nn.DataParallel(dis, device_ids=range(opt.ngpu))
    else:
        raise ValueError("Must run on gpus, ngpu > 0")
    if opt.starting_epoch > 0:
        gen.load_state_dict(torch.load('./{}/gen_epoch_{}.pth'.format(opt.out_f, opt.starting_epoch - 1)))
        dis.load_state_dict(torch.load('./{}/dis_epoch_{}.pth'.format(opt.out_f, opt.starting_epoch - 1)))
    return gen, dis

def get_loss():
    return loss_nll, loss_nll

def make_optimizer(model, beta1=0.9, beta2=0.999):
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(beta1, beta2))
    return optimizer

def make_dataset():
    def noise(x):
        return x + torch.FloatTensor(x.size()).uniform_(0, 1.0 / 128)
    if opt.dataset == "cifar10":
        trans = tfs.Compose([
            tfs.RandomCrop(opt.img_width, padding=4),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
            tfs.Lambda(noise)])
        data = CIFAR10(root=opt.root, train=True, download=False, transform=trans)
        loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    elif opt.dataset == "dog_and_cat_64":
        trans = tfs.Compose([
            tfs.RandomResizedCrop(opt.img_width, scale=(0.8, 0.9), ratio=(1.0, 1.0)),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
            tfs.Lambda(noise)])
        data = ImageFolder(opt.root, transform=trans)
        loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    elif opt.dataset == "dog_and_cat_128":
        trans = tfs.Compose([
            tfs.RandomResizedCrop(opt.img_width, scale=(0.8, 0.9), ratio=(1.0, 1.0)),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
            tfs.Lambda(noise)])
        data = ImageFolder(opt.root, transform=trans)
        loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    elif opt.dataset == "imagenet":
        trans = tfs.Compose([
            tfs.RandomResizedCrop(opt.img_width, scale=(0.8, 0.9), ratio=(1.0, 1.0)),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
            tfs.Lambda(noise)])
        data = ImageFolder(opt.root, transform=trans)
        loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    else:
        raise ValueError("Unknown dataset: {}".format(opt.dataset))
    return loader

def train():
    # models
    gen, dis = load_models()
    # optimizers
    opt_g, opt_d = make_optimizer(gen), make_optimizer(dis)
    # data
    train_loader = make_dataset()
    # buffer:
    # gaussian noise
    z = torch.FloatTensor(opt.batch_size, opt.nz).cuda()
    fixed_z = Variable(torch.FloatTensor(8 * 10, opt.nz).normal_(0, 1).cuda())
    # random label
    y_fake = torch.LongTensor(opt.batch_size).cuda()
    np_y = np.arange(10)
    np_y = np.repeat(np_y, 8)
    fixed_y_fake = Variable(torch.from_numpy(np_y).cuda())
    # fixed label
    zeros = Variable(torch.FloatTensor(opt.batch_size).fill_(0).cuda())
    ones = Variable(torch.FloatTensor(opt.batch_size).fill_(1).cuda())
    # loss
    Ld, Lg = get_loss()
    # start training
    for epoch in range(opt.starting_epoch, opt.starting_epoch + opt.max_epoch):
        for count, (x_real, y_real) in enumerate(train_loader):
            if count % opt.iter_d == 0:
                # update generator for every iter_d iterations
                gen.zero_grad()
                # sample noise
                z.normal_(0, 1)
                vz = Variable(z)
                y_fake.random_(0, to=opt.nclass)
                v_y_fake = Variable(y_fake)
                v_x_fake = gen(vz, y=v_y_fake)
                #ones.data.resize_(v_y_fake.size())
                v_x_fake_adv = v_x_fake
                #v_x_fake_adv = attack_FGSM(v_x_fake, ones, v_y_fake, dis, Lg)
                d_fake_bin, d_fake_multi = dis(v_x_fake_adv)
                ones.data.resize_as_(d_fake_bin.data)
                loss_g = Lg(d_fake_bin, ones, d_fake_multi, v_y_fake, lam=0.5)
                loss_g.backward()
                opt_g.step()
                print('[{}/{}][{}/{}][G_ITER] loss_g: {}'.format(epoch, opt.max_epoch-1, \
                        count+1, len(train_loader), loss_g.data[0]))
            # update discriminator
            dis.zero_grad()
            # feed real data
            x_real, y_real = x_real.cuda(), y_real.cuda()
            v_x_real, v_y_real = Variable(x_real), Variable(y_real)
            # find adversarial example
            ones.data.resize_(y_real.size())
            #v_x_real_adv = v_x_real
            v_x_real_adv = attack_Linf_PGD(v_x_real, ones, v_y_real, dis, Ld, opt.adv_steps, opt.epsilon)
            d_real_bin, d_real_multi = dis(v_x_real_adv)
            # accuracy for real images
            positive = torch.sum(d_real_bin.data > 0)
            _, idx = torch.max(d_real_multi.data, dim=1)
            correct_real = torch.sum(idx.eq(y_real))
            total_real = y_real.numel()
            # loss for real images
            loss_d_real = Ld(d_real_bin, ones, d_real_multi, v_y_real, lam=0.5)
            # feed fake data
            z.normal_(0, 1)
            y_fake.random_(0, to=opt.nclass)
            vz, v_y_fake = Variable(z), Variable(y_fake)
            with torch.no_grad():
                v_x_fake = gen(vz, y=v_y_fake)
            d_fake_bin, d_fake_multi = dis(v_x_fake.detach())
            # accuracy for fake images
            negative = torch.sum(d_fake_bin.data > 0)
            _, idx = torch.max(d_fake_multi.data, dim=1)
            correct_fake = torch.sum(idx.eq(y_fake))
            total_fake = y_fake.numel()
            # loss for fake images
            loss_d_fake = Ld(d_fake_bin, zeros, d_fake_multi, v_y_fake, lam=1)
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()
            print('[{}/{}][{}/{}][D_ITER] loss_d: {} acc_r: {}, acc_r@1: {}, acc_f: {}, acc_f@1: {}'.format(
                epoch, opt.max_epoch-1, count+1, len(train_loader), loss_d.data[0], positive / total_real,
                correct_real / total_real, negative / total_fake, correct_fake / total_fake))
        # generate samples
        with torch.no_grad():
            fixed_x_fake = gen(fixed_z, y=fixed_y_fake)
            fixed_x_fake.data.mul_(0.5).add_(0.5)
        x_real.mul_(0.5).add_(0.5)
        save_image(fixed_x_fake.data, './{}/sample_epoch_{}.png'.format(opt.out_f, epoch), nrow=8)
        save_image(x_real, './{}/real.png'.format(opt.out_f))
        # save model
        torch.save(dis.state_dict(), './{}/dis_epoch_{}.pth'.format(opt.out_f, epoch))
        torch.save(gen.state_dict(), './{}/gen_epoch_{}.pth'.format(opt.out_f, epoch))
        # change step size
        if (epoch + 1) % 50 == 0:
            opt.lr /= 2
            opt_g, opt_d = make_optimizer(gen), make_optimizer(dis)

if __name__ == "__main__":
    train()
