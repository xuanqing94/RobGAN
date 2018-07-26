#!/bin/bash


CUDA_VISIBLE_DEVICES=2,3,4,5 python finetune.py \
    --model resnet_128 \
    --netD ./ckpt.adv-5.128px/dis_epoch_119.pth \
    --netG ./ckpt.adv-5.128px/gen_epoch_119.pth \
    --ndf 64 \
    --ngf 64 \
    --nclass 143 \
    --dataset dog_and_cat_128 \
    --batch_size 128 \
    --root /nvme0/sngan_dog_cat \
    --img_width 128 \
    --steps 5 \
    --epsilon 0.03125 \
    --lam 0 \
    --lr 1.0e-3 \
    --ngpu 4 \
    --out_f ckpt.adv-5.128px.finetune \
    > >(tee log_finetune.txt) 2>error.txt
