#!/usr/bin/env python
# coding: utf-8


        # suppoprt to do batch accumulation for backprop with effectively larger batch size
        # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulationa
class Config(object):
    TRAIN_IMAGE_DIR = '../input/train_images'
    TRAIN_CSV = "../input/train_images.csv"
    TEST_CSV = "../input/test_images_2.csv"
    TEST_IMAGE_DIR = "../input/test_images_2"
    INPUT_DIR = "../input/"
    OUTPUT_DIR = f'../submission/sub40-2/models'
    #OUTPUT_DIR2 = f'../submission/sub35/models_with_pseudo'
    SUBMISSION = f"../submission/sub40-2"
    target_col='class_num'
    device='cuda:0'
    num_class=4
    
    DEBUG=False
    epochs=70
    model_arch = 'tf_efficientnet_b4_ns'
    img_size=380
    use_folds=list(range(14))
    seed= 2021
    seeds=list(range(1))
    #ARG_seeds=list(range(10))
    train_bs=16
    valid_bs=16
    
    T_0=10
    lr=1e-4
    min_lr=1e-7
    weight_decay=1e-6
    num_workers=2
    accum_iter = 2
    verbose_step=8
