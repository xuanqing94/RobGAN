for i in 7
do
    CUDA_VISIBLE_DEVICES=1 python eval_accuracy.py \
        --model resnet_imagenet \
        --model_in ./ckpt.adv-5.imagenet/dis_epoch_$i.pth \
        --ndf 64 \
        --nclass 1000 \
        --batch_size 300 \
        --root /data3/sngan_imagenet_val \
        --img_width 128 \
        --ngpu 1 \
        --workers 2 \
        > >(tee -a accuracy_imagenet.txt) 2>./error.log
done
