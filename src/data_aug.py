#!/usr/bin/python
# -*- coding: UTF-8 -*- 
# Author:   Hongwei Zhang
# FileName: data_aug.py
# Description:

from torchvision import transforms
import numpy as np
import random
import albumentations as A
from matplotlib import pyplot as plt

train_transformer1 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(45),
    transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
    transforms.ToTensor(),
])

train_transformer2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(45),
    transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
    # transforms.ColorJitter(brightness=0.4, contrast=0, saturation=0, hue=0),
    transforms.ToTensor(),
])


train_transformer3 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

def train_data_aug_offline(X_keyframes_train, Y_keyframes_seg_train, X_keyframes_name_train):
    print('Data Augumentation...')
    n_data = len(X_keyframes_name_train)

    X_keyframes_train_t = [];Y_keyframes_seg_train_t =[]; X_keyframes_name_train_t =[]

    seed = np.random.randint(2147483647)
    random.seed(seed)
    transform_topil = transforms.ToPILImage()

    transform_rot90 = transforms.RandomRotation(degrees=(90,90))
    transform_rot180 = transforms.RandomRotation(degrees=(180,180))
    transform_rot270 = transforms.RandomRotation(degrees=(270, 270))

    transform_zoom0_1 = A.Affine(scale=1.1, p=1)
    transform_zoom_0_1 = A.Affine(scale=0.9, p=1)

    transform_width0_1 = A.Affine(translate_percent={"x": 0.1, "y": 0}, p=1)
    transform_width_0_1 = A.Affine(translate_percent={"x": -0.1, "y": 0}, p=1)

    transform_height0_1 = A.Affine(translate_percent={"x": 0, "y": 0.1}, p=1)
    transform_height_0_1 = A.Affine(translate_percent={"x": 0, "y": -0.1}, p=1)

    transform_bright_0_2 = transforms.ColorJitter(brightness=(1.2,1.2))
    transform_bright0_2 = transforms.ColorJitter(brightness=(0.8,0.8))

    transform_contrast_0_2 = transforms.ColorJitter(contrast=(1.2,1.2))
    transform_contrast0_2 = transforms.ColorJitter(contrast=(0.8,0.8))

    transform_GaussianBlur3 = A.GaussianBlur(blur_limit=(3, 3), sigma_limit=0, p=1)
    transform_GaussianBlur5 = A.GaussianBlur(blur_limit=(5, 5), sigma_limit=0, p=1)

    transform_MotionBlur3 = A.MotionBlur(blur_limit=(3, 3), p=1)
    transform_MotionBlur5 = A.MotionBlur(blur_limit=(5, 5), p=1)

    transform_GaussNoise10 = A.GaussNoise(var_limit=(10.0, 10.0), mean=0, p=1)
    transform_GaussNoise30 = A.GaussNoise(var_limit=(30.0, 30.0), mean=0, p=1)

    for i in range(n_data):
        pics = [];labels=[];names=[]
        pic_orgin = X_keyframes_train[i]
        label_origin = Y_keyframes_seg_train[i]
        name_origin = X_keyframes_name_train[i][0:-4]

        pic = transform_topil(pic_orgin)
        label = transform_topil(label_origin)

        pics.append(pic_orgin)
        labels.append(label_origin)
        names.append(name_origin)
        pics.append(np.array(transform_rot90(pic)))
        labels.append(np.array(transform_rot90(label)))
        names.append(name_origin + '_rot90.jpg')
        pics.append(np.array(transform_rot180(pic)))
        labels.append(np.array(transform_rot180(label)))
        names.append(name_origin + '_rot180.jpg')
        pics.append(np.array(transform_rot270(pic)))
        labels.append(np.array(transform_rot270(label)))
        names.append(name_origin + '_rot270.jpg')
        pics.append(transform_zoom0_1(image=pic_orgin, mask=label_origin)['image'])
        labels.append(transform_zoom0_1(image=pic_orgin, mask=label_origin)['mask'])
        names.append(name_origin + '_zoom10.jpg')
        pics.append(transform_zoom_0_1(image=pic_orgin, mask=label_origin)['image'])
        labels.append(transform_zoom_0_1(image=pic_orgin, mask=label_origin)['mask'])
        names.append(name_origin + '_zoom_10.jpg')
        pics.append(transform_width0_1(image=pic_orgin, mask=label_origin)['image'])
        labels.append(transform_width0_1(image=pic_orgin, mask=label_origin)['mask'])
        names.append(name_origin + '_width10.jpg')
        pics.append(transform_width_0_1(image=pic_orgin, mask=label_origin)['image'])
        labels.append(transform_width_0_1(image=pic_orgin, mask=label_origin)['mask'])
        names.append(name_origin + '_width_10.jpg')
        pics.append(transform_height0_1(image=pic_orgin, mask=label_origin)['image'])
        labels.append(transform_height0_1(image=pic_orgin, mask=label_origin)['mask'])
        names.append(name_origin + '_height10.jpg')
        pics.append(transform_height_0_1(image=pic_orgin, mask=label_origin)['image'])
        labels.append(transform_height_0_1(image=pic_orgin, mask=label_origin)['mask'])
        names.append(name_origin + '_height_10.jpg')
        pics.append(np.array(transform_bright0_2(pic)))
        labels.append(label_origin)
        names.append(name_origin + '_bright20.jpg')
        pics.append(np.array(transform_bright_0_2(pic)))
        labels.append(label_origin)
        names.append(name_origin + '_bright_20.jpg')
        pics.append(np.array(transform_contrast0_2(pic)))
        labels.append(label_origin)
        names.append(name_origin + '_contrast20.jpg')
        pics.append(np.array(transform_contrast_0_2(pic)))
        labels.append(label_origin)
        names.append(name_origin + '_contrast_20.jpg')
        pics.append(transform_GaussianBlur3(image=pic_orgin, mask=label_origin)['image'])
        labels.append(transform_GaussianBlur3(image=pic_orgin, mask=label_origin)['mask'])
        names.append(name_origin + '_GaussianBlur3.jpg')
        pics.append(transform_GaussianBlur5(image=pic_orgin, mask=label_origin)['image'])
        labels.append(transform_GaussianBlur5(image=pic_orgin, mask=label_origin)['mask'])
        names.append(name_origin + '_GaussianBlur5.jpg')
        pics.append(transform_MotionBlur3(image=pic_orgin, mask=label_origin)['image'])
        labels.append(transform_MotionBlur3(image=pic_orgin, mask=label_origin)['mask'])
        names.append(name_origin + '_MotionBlur3.jpg')
        pics.append(transform_MotionBlur5(image=pic_orgin, mask=label_origin)['image'])
        labels.append(transform_MotionBlur5(image=pic_orgin, mask=label_origin)['mask'])
        names.append(name_origin + '_MotionBlur5.jpg')
        pics.append(transform_GaussNoise10(image=pic_orgin, mask=label_origin)['image'])
        labels.append(transform_GaussNoise10(image=pic_orgin, mask=label_origin)['mask'])
        names.append(name_origin + '_GaussNoise10.jpg')
        pics.append(transform_GaussNoise30(image=pic_orgin, mask=label_origin)['image'])
        labels.append(transform_GaussNoise30(image=pic_orgin, mask=label_origin)['mask'])
        names.append(name_origin + '_GaussNoise30.jpg')

        X_keyframes_train_t.extend(pics)
        Y_keyframes_seg_train_t.extend(labels)
        X_keyframes_name_train_t.extend(names)

    print('Data Augmentation End')
    return X_keyframes_train_t,Y_keyframes_seg_train_t, X_keyframes_name_train_t

def shuffle_data(X_keyframes, Y_keyframes_seg, X_keyframes_name):
    data_num = len(X_keyframes_name)

    np.random.seed(888)
    shuffle_indices = np.random.permutation(np.arange(data_num))

    X_keyframes = [X_keyframes[i] for i in shuffle_indices]
    Y_keyframes_seg = [Y_keyframes_seg[i] for i in shuffle_indices]
    X_keyframes_name = [X_keyframes_name[i] for i in shuffle_indices]

    return X_keyframes, Y_keyframes_seg, X_keyframes_name