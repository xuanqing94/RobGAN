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
    --root /data1/xqliu/sngan_data \
    --img_width 128 \
    --iter_d 5 \
    --out_f ckpt.adv-5.128px-imagenet \
    --ngpu 5 \
    --starting_epoch 0 \
    --max_epoch 200 \
    --lr 0.0002 \
    --adv_steps 5 \
    --epsilon 0.0625 \
    --our_loss # Our ACGAN
