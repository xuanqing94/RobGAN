#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ./train_cifar.py \
    --model resnet_32 \
    --dataset cifar10 \
    --nz 128 \
    --ngf 64 \
    --ndf 64 \
    --nclass 10 \
    --batch_size 64 \
    --start_width 4 \
    --root ~/data/cifar10-py \
    --img_width 32 \
    --iter_d 1 \
    --out_f ckpt.acgan \
    --ngpu 1 \
    --max_epoch 100 \
    --lr 0.0002 \
    --adv_steps 5 \
    --epsilon 0.07
