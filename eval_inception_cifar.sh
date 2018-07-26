#!/bin/bash

#for i in {101..119}
for i in 41
do
    CUDA_VISIBLE_DEVICES=1,2 python ./eval_inception_cifar.py \
        --model_in ./ckpt.acgan/gen_epoch_$i.pth \
        --nz 128 \
        --ngf 64 \
        --nimgs 50000 \
        --batch_size 100 \
        --start_width 4 \
        --splits 10 \
        --ngpu 2 #| tee -a score_adv_32px.txt
done
