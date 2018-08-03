#!/usr/bin/env python
import sys
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.transforms as tfs
from torchvision.datasets import ImageFolder, CIFAR10
from torch.utils.data import DataLoader
from miscs.pgd import attack_label_Linf_PGD

cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--netD', type=str, required=True)
parser.add_argument('--netG', type=str, required=True)
parser.add_argument('--ndf', type=int, required=True)
parser.add_argument('--ngf', type=int, required=True)
parser.add_argument('--nclass', type=int, required=True)
parser.add_argument('--nz', type=int, default=128)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--root', type=str, required=True)
parser.add_argument('--start_width', type=int, default=4)
parser.add_argument('--img_width', type=int, required=True)
parser.add_argument('--steps', type=int, required=True)
parser.add_argument('--epsilon', type=float, required=True)
parser.add_argument('--lam', type=float, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--ngpu', type=int, required=True)
parser.add_argument('--workers', type=int, default=3)
parser.add_argument('--out_f', type=str, required=True)
opt = parser.parse_args()


def load_models():
    if opt.model == "resnet_32":
        from gen_models.resnet_32 import ResNetGenerator
        from dis_models.preact_resnet import PreActResNet18
        gen = ResNetGenerator(ch=opt.ngf, dim_z=opt.nz, bottom_width=opt.start_width, n_classes=opt.nclass)
        dis = PreActResNet18()
    elif opt.model == "resnet_64":
        from gen_models.resnet_64 import ResNetGenerator
        from dis_models.resnet_64 import ResNetAC
        gen = ResNetGenerator(ch=opt.ngf, dim_z=opt.nz, bottom_width=opt.start_width, n_classes=opt.nclass)
        dis = ResNetAC(ch=opt.ndf, n_classes=opt.nclass)
    elif opt.model == "resnet_128":
        from gen_models.resnet_small import ResNetGenerator
        from dis_models.resnet_small import ResNetAC
        gen = ResNetGenerator(ch=opt.ngf, dim_z=opt.nz, bottom_width=opt.start_width, n_classes=opt.nclass)
        dis = ResNetAC(ch=opt.ndf, n_classes=opt.nclass, bn=True) #XXX here we choose bn=True, because of improper initialization
    elif opt.model == "resnet_imagenet":
        from gen_models.resnet import ResNetGenerator
        from dis_models.resnet import ResNetAC
        gen = ResNetGenerator(ch=opt.ngf, dim_z=opt.nz, bottom_width=opt.start_width, n_classes=opt.nclass)
        dis = ResNetAC(ch=opt.ndf, n_classes=opt.nclass)
    else:
        raise ValueError(f"Unknown model name: {opt.model}")
    if opt.ngpu > 0:
        gen, dis = gen.cuda(), dis.cuda()
        gen, dis = torch.nn.DataParallel(gen, device_ids=range(opt.ngpu)), \
                torch.nn.DataParallel(dis, device_ids=range(opt.ngpu))
    else:
        raise ValueError("Must run on gpus, ngpu > 0")
    gen.load_state_dict(torch.load(opt.netG))
    dis.load_state_dict(torch.load(opt.netD))
    return gen, dis

def make_dataset():
    if opt.dataset == "cifar10":
        trans = tfs.Compose([
            tfs.RandomCrop(opt.img_width, padding=4),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
            ])
        data = CIFAR10(root=opt.root, train=True, download=False, transform=trans)
        data_test = CIFAR10(root=opt.root, train=False, download=False, transform=trans)
        loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
        loader_test = DataLoader(data_test, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    elif opt.dataset == "dog_and_cat_64":
        trans = tfs.Compose([
            tfs.RandomResizedCrop(opt.img_width, scale=(0.8, 0.9), ratio=(1.0, 1.0)),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
            ])
        data = ImageFolder(opt.root, transform=trans)
        data_test = ImageFolder("/data3/sngan_dog_cat_val", transform=trans)
        loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
        loader_test = DataLoader(data_test, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    elif opt.dataset == "dog_and_cat_128":
        trans = tfs.Compose([
            tfs.RandomResizedCrop(opt.img_width, scale=(0.8, 0.9), ratio=(1.0, 1.0)),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
            ])
        data = ImageFolder(opt.root, transform=trans)
        data_test = ImageFolder("/nvme0/sngan_dog_cat_val", transform=trans)
        loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
        loader_test = DataLoader(data_test, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    elif opt.dataset == "imagenet":
        trans = tfs.Compose([
            tfs.RandomResizedCrop(opt.img_width, scale=(0.8, 0.9), ratio=(1.0, 1.0)),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            tfs.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
            ])
        data = ImageFolder(opt.root, transform=trans)
        loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    else:
        raise ValueError(f"Unknown dataset: {opt.dataset}")
    return loader, loader_test

def test_acc(loader_test, dis):
    total = 0
    correct_label = 0
    for i, (x_real, y_real) in enumerate(loader_test):
        if i == 100:
            break
        x_real, y_real = x_real.cuda(), y_real.cuda()
        v_y_real, v_x_real = Variable(y_real), Variable(x_real)
        adv_input = attack_label_Linf_PGD(v_x_real, v_y_real, dis, opt.steps * 4, opt.epsilon)
        with torch.no_grad():
            _, d_multi = dis(adv_input)
        _, idx = torch.max(d_multi.data, dim=1)
        label_correct = idx.eq(y_real)
        correct_label += torch.sum(label_correct).item()
        total += y_real.numel()
    print(f'test_acc: {correct_label/total}')

def get_optimizer(parameters):
    #optimizer = torch.optim.SGD(dis.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5.0e-4)
    return torch.optim.Adam(parameters, lr=opt.lr)

def main():
    # model
    gen, dis = load_models()
    # data
    loader, loader_test = make_dataset()
    # optimizer
    optimizer = get_optimizer(dis.parameters())
    # loss function
    loss_f = nn.CrossEntropyLoss()
    # buffer
    noise = torch.FloatTensor(opt.batch_size, opt.nz).zero_().cuda()
    noise_v = Variable(noise)
    noise_y = torch.LongTensor(opt.batch_size).zero_().cuda()
    noise_y_v = Variable(noise_y)
    epochs = [20, 20, 10, 10]
    accumulate = 0
    for stage in epochs:
        for _ in range(stage):
            accumulate += 1
            for it, (x, y) in enumerate(loader):
                # feed real images
                x, y = x.cuda(), y.cuda()
                vx_real, vy = Variable(x), Variable(y)
                vx_real_adv = attack_label_Linf_PGD(vx_real, vy, dis,
                        opt.steps, opt.epsilon)
                _, output_real = dis(vx_real_adv)
                loss_real = loss_f(output_real, vy)

                # feed fake images
                if opt.lam > 0:
                    noise_v.normal_(0, 1)
                    noise_y.random_(0, to=opt.nclass)
                    with torch.no_grad():
                        vx_fake = gen(noise_v, noise_y_v)
                    vx_fake_adv = attack_label_Linf_PGD(vx_fake, noise_y_v,
                             dis, opt.steps, opt.epsilon)
                    _, output_fake = dis(vx_fake_adv)
                    loss_fake = loss_f(output_fake, noise_y_v)
                    # combined loss
                    loss_total = loss_real + opt.lam * loss_fake
                else:
                    loss_total = loss_real
                dis.zero_grad()
                loss_total.backward()
                optimizer.step()
                # accuracy on real / fake
                _, idx = torch.max(output_real, dim=1)
                correct_real = torch.sum(y.eq(idx.data)).item()
                accuracy_real = correct_real / y.numel()
                if opt.lam > 0:
                    _, idx = torch.max(output_fake, dim=1)
                    correct_fake = torch.sum(noise_y.eq(idx.data)).item()
                    accuracy_fake = correct_fake / noise_y.numel()
                    print(f'[{accumulate}][{it}/{len(loader)}] acc_real: {accuracy_real}, acc_fake: {accuracy_fake}')
                else:
                    print(f'[{accumulate}][{it}/{len(loader)}] acc_real: {accuracy_real}, acc_fake: NA')
                sys.stdout.flush()
            # test
            test_acc(loader_test, dis)
            # save model
            torch.save(dis.state_dict(), f'./{opt.out_f}/dis_finetune_{accumulate}.pth')
        opt.lr /= 10
        #optimizer = torch.optim.SGD(dis.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5.0e-4)
        optimizer = get_optimizer(dis.parameters())

if __name__ == "__main__":
    main()
