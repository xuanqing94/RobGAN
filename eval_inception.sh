#!/bin/bash

for i in {120..200}; do
    CUDA_VISIBLE_DEVICES=0,1,4,5 python ./eval_inception.py \
        --model resnet_64 \
        --model_in ./ckpt.adv-0.64px-acloss/gen_epoch_$i.pth \
        --nz 128 \
        --ngf 64 \
        --nclass 143 \
        --nimgs 50000 \
        --batch_size 100 \
        --start_width 4 \
        --splits 10 \
        --ngpu 4 2>error.txt
done
