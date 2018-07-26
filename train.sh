#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3,4,5 python ./train.py \
    --model resnet_imagenet \
    --nz 128 \
    --ngf 64 \
    --ndf 64 \
    --nclass 1000 \
    --batch_size 64 \
    --start_width 4 \
    --dataset imagenet \
    --root /nvme0/sngan_data \
    --img_width 128 \
    --iter_d 15 \
    --out_f ckpt.adv-10.imagenet \
    --ngpu 5 \
    --starting_epoch 0 \
    --max_epoch 80 \
    --lr 0.0002 \
    --adv_steps 10 \
    --epsilon 0.063
