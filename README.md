# AdvGAN
*From Adversarial Training to Generative Adversarial Networks*

**Still under construction**

## Requirements
Below is my running environment:

+ Python==3.6 + PyTorch>=0.4.0

## File structure

+ `train.[py, sh]`: GAN training (Step 1)
+ `fintune.[py, sh]`: Fine-tuning (Step 2)
+ `eval_inception.[py, sh]`: Evaluate the inception score
+ `acc_under_attack.[py, sh]`: Evaluate the accuracy under PGD attack
+ `/dis_models`: discriminators
+ `/gen_models`: generators
+ `/layers`: customized layers
+ `/miscs`: loss function, pgd-attack, etc.

## Step 0. Data Preparation
Follow the [SNGAN-projection](https://github.com/pfnet-research/sngan_projection#preprocess-dataset) steps to download and pre-process data.

Hereafter, I assume the 1000-class ImageNet data is stored in `/data1/sngan_data`, 143-class ImageNet data is stored in `/data1/sngan_dog_cat` and `/data1/sngan_dog_cat_val` (for validation).

## Step 1. Co-training the generator and discriminator

