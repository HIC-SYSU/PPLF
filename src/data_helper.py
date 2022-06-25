#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Author:   Hongwei Zhang
# FileName: data_helper.py
# Description:

import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import matplotlib.pyplot as plt
from data_aug import *

class DSA_keyframe(Dataset):
    '''处理dsa数据，包括dsa图片、标签、图片名称
    parameters
        data_path: str, 数据路径
        is_train: bool, 训练集 or 测试集
        online_aug: bool, 在线数据增强 or not
        offline_aug: bool, 离线数据增强 or not
        fold_num: int, 十折交叉的fold_num折
    '''
    def __init__(self, data_path, is_train_val_test='0', online_aug=False, offline_aug=False, folds_all=None, fold_th=None):
        self.is_train_val_test = is_train_val_test
        self.online_aug = online_aug
        self.offline_aug = offline_aug

        path1 = data_path

        # 加载所有数据
        self.X_dsa_keyframe = pickle.load(open(path1 + 'X_dsa_keyframe.pk', 'rb'))
        self.Y_dsa_keyframe_seg = pickle.load(open(path1 + 'Y_dsa_keyframe_seg.pk', 'rb'))
        self.X_dsa_keyframe_name = pickle.load(open(path1 + 'X_dsa_keyframe_name.pk', 'rb'))

        assert len(self.X_dsa_keyframe) == len(self.Y_dsa_keyframe_seg) == len(self.X_dsa_keyframe_name)

        #按x折交叉将数据分为训练和测试
        self.X_keyframes_train, self.Y_keyframes_seg_train, self.X_keyframes_name_train, self.X_keyframes_val, self.Y_keyframes_seg_val, self.X_keyframes_name_val, self.X_keyframes_test, self.Y_keyframes_seg_test, self.X_keyframes_name_test = spilt_data_x_fold(
            self.X_dsa_keyframe, self.Y_dsa_keyframe_seg, self.X_dsa_keyframe_name, folds_all=folds_all, fold_th = fold_th)

        self.train_true_len = len(self.X_keyframes_train)

        if self.offline_aug == True:
            '''将训练集数据增强
            '''
            self.X_keyframes_train, self.Y_keyframes_seg_train, self.X_keyframes_name_train= train_data_aug_offline(self.X_keyframes_train, self.Y_keyframes_seg_train, self.X_keyframes_name_train)

    def __getitem__(self, index):
        if self.is_train_val_test == '1':
            patient = self.X_keyframes_train[index]
            patient_seg_label = self.Y_keyframes_seg_train[index]
            patient_name = self.X_keyframes_name_train[index]

            if self.online_aug == True:
                patient, patient_seg_label = online_aug(patient, patient_seg_label, patient_name)
            else:
                patient = train_transformer3(patient)
                patient_seg_label = train_transformer3(patient_seg_label)
        elif self.is_train_val_test == '2':
            patient = self.X_keyframes_val[index]
            patient_seg_label = self.Y_keyframes_seg_val[index]
            patient_name = self.X_keyframes_name_val[index]

            patient = train_transformer3(patient)
            patient_seg_label = train_transformer3(patient_seg_label)
        elif self.is_train_val_test == '3':
            patient = self.X_keyframes_test[index]
            patient_seg_label = self.Y_keyframes_seg_test[index]
            patient_name = self.X_keyframes_name_test[index]

            patient = train_transformer3(patient)
            patient_seg_label = train_transformer3(patient_seg_label)

        patient = torch.cat((patient,patient,patient), dim=0)
        sample = {'patient': patient, 'seg_label': patient_seg_label, 'patient_name': patient_name}
        return sample

    def __len__(self):
        if self.is_train_val_test == '1':
            return len(self.X_keyframes_name_train)
        elif self.is_train_val_test =='2':
            return len(self.X_keyframes_name_val)
        elif self.is_train_val_test =='3':
            return len(self.X_keyframes_name_test)

def online_aug(patient, patient_seg_label, patient_name):
    seed = np.random.randint(2147483647)
    random.seed(seed)
    patient = train_transformer1(patient)
    random.seed(seed)
    patient_seg_label_t = train_transformer2(patient_seg_label)
    patient_seg_label = torch.gt(patient_seg_label_t, 0.49).type(torch.float32)

    return patient, patient_seg_label

def spilt_data_x_fold(X_keyframes, Y_keyframes_seg, X_keyframes_name, folds_all=10, fold_th = 1):
    assert len(X_keyframes) == len(Y_keyframes_seg) == len(X_keyframes_name)

    np.random.seed(888)
    data_num = len(X_keyframes_name)
    shuffle_indices = np.random.permutation(np.arange(data_num))
    train_indice, val_indice, test_indice = valid(shuffle_indices, folds_all=folds_all, i = fold_th)

    X_keyframes_train = [X_keyframes[i] for i in train_indice]
    Y_keyframes_seg_train = [Y_keyframes_seg[i] for i in train_indice]
    X_keyframes_name_train = [X_keyframes_name[i] for i in train_indice]

    X_keyframes_val = [X_keyframes[i] for i in val_indice]
    Y_keyframes_seg_val = [Y_keyframes_seg[i] for i in val_indice]
    X_keyframes_name_val = [X_keyframes_name[i] for i in val_indice]

    X_keyframes_test = [X_keyframes[i] for i in test_indice]
    Y_keyframes_seg_test = [Y_keyframes_seg[i] for i in test_indice]
    X_keyframes_name_test = [X_keyframes_name[i] for i in test_indice]

    return X_keyframes_train, Y_keyframes_seg_train, X_keyframes_name_train, X_keyframes_val, Y_keyframes_seg_val, X_keyframes_name_val, X_keyframes_test, Y_keyframes_seg_test, X_keyframes_name_test

def valid(shuffle_indices, folds_all=5, i=1):
    '''将所有数据序号打乱，按x折交叉，返回第i折的训练和测试序号
    param
        shuffle_indices: list, 所有数据的序号
        folds_all: int, default 5, 总的折数
        param i: int, 第i折
    return
        train: list, 训练数据的序号
        test: list, 测试数据的序号
    '''
    a = np.linspace(0, len(shuffle_indices), folds_all+1, dtype=int)

    if i == folds_all:
        test = shuffle_indices[a[i - 1]:]
        val = shuffle_indices[:a[1]]
        train = shuffle_indices[a[1]:a[i - 1]]

    elif i == 1:
        test = shuffle_indices[:a[i]]
        val = shuffle_indices[a[i]:a[i+1]]
        train = shuffle_indices[a[i+1]:]

    else:
        test = shuffle_indices[a[i - 1]:a[i]]
        val = shuffle_indices[a[i]:a[i+1]]
        train = list(shuffle_indices[a[0]:a[i - 1]]) + list(shuffle_indices[a[i+1]:])

    return train, val, test

def visualize(image):
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    all_data_path = '../../Data/dsa_256/data_all/'
    online_aug = True;offline_aug=True
    dataset_train = DSA_keyframe(all_data_path, is_train_val_test='1',online_aug=online_aug,offline_aug=offline_aug, folds_all=5, fold_th=1)
    dataset_val = DSA_keyframe(all_data_path, is_train_val_test='2',online_aug=False,offline_aug=False, folds_all=5, fold_th=1)
    dataset_test = DSA_keyframe(all_data_path, is_train_val_test='3',online_aug=False,offline_aug=False, folds_all=5, fold_th=1)

    train_loader = DataLoader(dataset_train, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)


    for batch in train_loader:
        a = batch
        print(1)

