#!/usr/bin/env python

import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.models import inception_v3
from miscs.inception_score import inception_score as score

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--model_in', type=str, required=True)
parser.add_argument('--nz', type=int, required=True)
parser.add_argument('--ngf', type=int, required=True)
parser.add_argument('--nclass', type=int, required=True)
parser.add_argument('--nimgs', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--start_width', type=int, required=True)
parser.add_argument('--splits', type=int, required=True)
parser.add_argument('--ngpu', type=int, required=True)
opt = parser.parse_args()

assert opt.nimgs % opt.splits == 0, "ERR: opt.nimgs must be divided by opt.splits"
assert (opt.nimgs // opt.splits) % opt.batch_size == 0, "ERR: opt.nimgs//opt.splits \
        must be divided by opt.batch_size"

def load_model():
    if opt.model == "resnet_32":
        from gen_models.resnet_32 import ResNetGenerator
        gen = ResNetGenerator(ch=opt.ngf, dim_z=opt.nz, bottom_width=opt.start_width, n_classes=opt.nclass)
    elif opt.model == "resnet_64":
        from gen_models.resnet_64 import ResNetGenerator
        gen = ResNetGenerator(ch=opt.ngf, dim_z=opt.nz, bottom_width=opt.start_width, n_classes=opt.nclass)
    elif opt.model == "resnet_128":
        from gen_models.resnet_small import ResNetGenerator
        gen = ResNetGenerator(ch=opt.ngf, dim_z=opt.nz, bottom_width=opt.start_width, n_classes=opt.nclass)
    else:
        raise ValueError(f"Unknown model name: {opt.model}")
    if opt.ngpu > 0:
        gen = gen.cuda()
        gen = torch.nn.DataParallel(gen, device_ids=range(opt.ngpu))
    else:
        raise ValueError("Must run on gpus, ngpu > 0")
    gen.load_state_dict(torch.load(opt.model_in))
    return gen

def load_inception():
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.cuda()
    inception_model = torch.nn.DataParallel(inception_model, \
            device_ids=range(opt.ngpu))
    inception_model.eval()
    return inception_model

def gen_imgs():
    gen = load_model()
    # buffer:
    # gaussian noise
    z = torch.FloatTensor(opt.batch_size, opt.nz).cuda()
    # random label
    y_fake = torch.LongTensor(opt.batch_size).cuda()
    imgs = []
    with torch.no_grad():
        for i in range(0, opt.nimgs, opt.batch_size):
            z.normal_(0, 1)
            y_fake.random_(0, to=opt.nclass)
            v_z = Variable(z)
            v_y_fake = Variable(y_fake)
            x_fake = gen(v_z, y=v_y_fake)
            x = x_fake.data.cpu().numpy()
            imgs.append(x)
    imgs = np.asarray(imgs, dtype=np.float32)
    nb, b, c, h, w = imgs.shape
    imgs = imgs.reshape((nb * b, c, h, w))
    return imgs, (h, w) != (299, 299)

def calc_inception():
    imgs, resize = gen_imgs()
    model = load_inception()
    mean_score, std_score = score(model, imgs, opt.batch_size, \
            resize, opt.splits)
    return mean_score, std_score

def main():
    mean, std = calc_inception()
    print(f"Mean: {mean}, Std: {std}")

if __name__ == "__main__":
    main()
