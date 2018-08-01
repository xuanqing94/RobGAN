#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3,4,5 python ./train.py \
    --model resnet_64 \
    --nz 128 \
    --ngf 64 \
    --ndf 64 \
    --nclass 143 \
    --batch_size 64 \
    --start_width 4 \
    --dataset dog_and_cat_64 \
    --root /data1/xqliu/sngan_dog_cat \
    --img_width 64 \
    --iter_d 5 \
    --out_f ckpt.adv-5.64px-acloss \
    --ngpu 5 \
    --starting_epoch 80 \
    --max_epoch 200 \
    --lr 0.0002 \
    --adv_steps 5 \
    --epsilon 0.03125 \
    #--our_loss # Our ACGAN
