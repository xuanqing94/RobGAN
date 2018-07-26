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
from miscs.pgd import attack_Linf_PGD_bin
from miscs.loss import *

parser = argparse.ArgumentParser()
parser.add_argument('--nz', type=int, required=True)
parser.add_argument('--ngf', type=int, required=True)
parser.add_argument('--ndf', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--start_width', type=int, required=True)
parser.add_argument('--root', type=str, required=True)
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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        if m.weight is not None:
            m.weight.data.fill_(1.0)
        if m.bias is not None:
            m.bias.data.fill_(0)

def load_models():
    #from gen_models.cnn_32 import Generator
    #from dis_models.cnn_32 import Discriminator

   gen = Generator()
    dis = Discriminator()
    gen.apply(weights_init)
    dis.apply(weights_init)
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
    return loss_bin, loss_bin

def make_optimizer(model, beta1=0.5, beta2=0.999):
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(beta1, beta2))
    #optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)
    return optimizer

def make_dataset():
    #def noise(x):
    #    return x + torch.FloatTensor(x.size()).uniform_(0, 1.0 / 128)
    trans = tfs.Compose([
        #tfs.RandomCrop(32, padding=4),
        #tfs.RandomHorizontalFlip(),
        tfs.ToTensor(),
        tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
    data = CIFAR10(root=opt.root, train=True, download=False, transform=trans)
    loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
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
    # fixed label
    zeros = Variable(torch.FloatTensor(opt.batch_size).fill_(0).cuda())
    ones = Variable(torch.FloatTensor(opt.batch_size).fill_(1).cuda())
    # loss
    Ld, Lg = get_loss()
    # start training
    for epoch in range(opt.starting_epoch, opt.starting_epoch + opt.max_epoch):
        for count, (x_real, y_real) in enumerate(train_loader):
            if count % opt.iter_d == 0:
            #while acc_fake < 0.5:
                # update generator for every iter_d iterations
                gen.zero_grad()
                # sample noise
                z.normal_(0, 1)
                vz = Variable(z)
                v_x_fake = gen(vz)
                d_fake_bin = dis(v_x_fake)
                ones.data.resize_as_(d_fake_bin.data)
                loss_g = Lg(d_fake_bin, ones)
                loss_g.backward()
                opt_g.step()
                print('[{}/{}][{}/{}][G_ITER] loss_g: {}'.format(epoch, opt.max_epoch-1, count+1,
                    len(train_loader), loss_g.data[0]))
            # update discriminator
            dis.zero_grad()
            # feed real data
            x_real = x_real.cuda()
            v_x_real = Variable(x_real)
            # find adversarial example
            ones.data.resize_(y_real.size())
            v_x_real_adv = attack_Linf_PGD_bin(v_x_real, ones, dis, Ld, opt.adv_steps, opt.epsilon)
            d_real_bin = dis(v_x_real_adv)
            # loss for real images
            loss_d_real = Ld(d_real_bin, ones)
            # feed fake data
            z.normal_(0, 1)
            vz = Variable(z)
            with torch.no_grad():
                v_x_fake = gen(vz)
            d_fake_bin = dis(v_x_fake.detach())
            # loss for fake images
            loss_d_fake = Ld(d_fake_bin, zeros)
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()
            print('[{}/{}][{}/{}][D_ITER] loss_d: {}'.format(epoch, opt.max_epoch-1,
                count+1, len(train_loader), loss_d.data[0]))
        # generate samples
        with torch.no_grad():
            fixed_x_fake = gen(fixed_z)
            fixed_x_fake.data.mul_(0.5).add_(0.5)
        x_real.mul_(0.5).add_(0.5)
        save_image(fixed_x_fake.data, './{}/sample_epoch_{}.png'.format(opt.out_f, epoch), nrow=8)
        save_image(x_real, './{}/real.png'.format(opt.out_f))
        # save model
        torch.save(dis.state_dict(), './{}/dis_epoch_{}.pth'.format(opt.out_f, epoch))
        torch.save(gen.state_dict(), './{}/gen_epoch_{}.pth'.format(opt.out_f, epoch))
        # change step size
        #if (epoch + 1) % 60 == 0:
        #    opt.lr /= 2
        #    opt_g, opt_d = make_optimizer(gen), make_optimizer(dis)

if __name__ == "__main__":
    train()
