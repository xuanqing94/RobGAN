#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python acc_under_attack.py \
    --model resnet_128 \
    --model_in ./ckpt.adv-5.128px.finetune/dis_finetune_6.pth \
    --ndf 64 \
    --nclass 143 \
    --dataset dog_and_cat_128  \
    --root /data3/sngan_dog_cat_val \
    --img_width 128 \
    --steps 20 \
    --epsilon 0,0.004,0.01,0.016,0.02,0.024,0.03,0.04 \
    --ngpu 6 \
    --workers 3
