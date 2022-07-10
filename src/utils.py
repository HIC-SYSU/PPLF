#!/usr/bin/python
# -*- coding: UTF-8 -*- 
# Author:   Hongwei Zhang
# FileName: utils.py
# Description:
import argparse
import torch
import numpy as np
import random
import torchvision.utils as vutils
import logging

def build_optimizer(optim, model, lr, weight_decay, momentum, nesterov=False):
    if optim == 'Sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
    if optim == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    if optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer

def get_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of PPLF for DSA main coronary segmentation")
    parser.add_argument('-i', '--info_help', dest='info_help', type=str, default='None',
                        help='train information')
    parser.add_argument('-e','--epochs', dest='epochs', type=int, default=2,
                        help='Number of epochs')
    parser.add_argument('-b','--batch_size', dest='batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('-l','--base_lr', dest='base_lr',type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('-g','--gpu', dest='gpu_num', type=str, default='2',
                        help='gpu to use')
    parser.add_argument('-f','--fold_th', dest='fold_th', type=int, default=1,
                        help='交叉验证的第x折')
    parser.add_argument('-fs','--folds_all', dest='folds_all', type=int, default=10,
                        help='交叉验证的总折数')
    parser.add_argument('-d','--data_path', dest='data_path',type=str, default='../../../Data/',
                        help='data_path')
    parser.add_argument('-ds','--data_size', dest='data_size',type=str, default='dsa_compare_256/',
                        help='数据尺度, dsa_256/ or dsa_512/')
    parser.add_argument('-dv','--data_vessel', dest='data_vessel',type=str, default='data_all/',
                        help='data_all')
    parser.add_argument('-a','--aug', dest='aug', action='store_true', default=False,
                        help='aug or not')
    parser.add_argument('--save_model', dest='save_model', action='store_false', default=True,
                        help='save_model')
    parser.add_argument('--save_pic_val', dest='save_pic_val', action='store_false', default=True,
                        help='save_pic_val')
    parser.add_argument('--save_pic_test', dest='save_pic_test', action='store_false', default=True,
                        help='save_pic_test')
    parser.add_argument('--save_pic_train', dest='save_pic_train', action='store_false', default=True,
                        help='save_pic_train')
    return parser

def color_predict_save(imgs, labels, preds, imgs_name, save_path):
    imgs_num = len(imgs)
    for i in range(imgs_num):
        img = torch.unsqueeze(imgs[i], dim=0)
        label = torch.unsqueeze(labels[i], dim=0)
        pred = torch.unsqueeze(preds[i], dim=0)
        img_name = imgs_name[i] if imgs_name[i].endswith('.bmp') else imgs_name[i]+'.bmp'

        color_map = torch.cat((label, 1 - (1 - label * (1 - pred)) * (1 - (1 - label) * pred), label - label), dim=1)
        color_pred =  img*(1-label)*(1-pred) + color_map

        pred_3ch = torch.cat((pred, pred, pred), dim=1)
        label_3ch = torch.cat((label, label, label), dim=1)

        pic_save = torch.cat((img, color_pred, pred_3ch, label_3ch),dim=0)

        vutils.save_image(pic_save, save_path+img_name, nrow=2, normalize=False, padding=0)

def logging_information(log, args, save_path, n_train_true, n_train, n_val):
    log.logger.info(f'''Training Information:
        Content:        {args.info_help}

        Fold:           {str(args.fold_th)+'/'+str(args.folds_all)}
        Epochs:         {args.epochs}
        Batch size:     {args.batch_size}
        Batch nums:     {n_train/args.batch_size}
        Learning rate:  {args.base_lr}
        Lr_step:        {args.step_size}
        Lr_gamma:       {args.gamma}
        Device:         {'gpu' + args.gpu_num}
        
        Training size:  {str(n_train_true) + ' -> ' + str(int(n_train))}
        Testing size:   {n_val}
        Data path:      {args.data_path + args.data_size + args.data_vessel}
        
        Save path:      {save_path}
        Save_model:     {args.save_model}
        Save_pic_val:  {args.save_pic_val}
        Save_pic_train: {args.save_pic_train}

        Model:          {args.model_name}  
        Optimizer:      {args.optim}
        Weigt_decay:    {args.weight_decay}
        Momentum:       {args.momentum}
    ''')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    } 

    def __init__(self, filename, level='info', when='D',interval=7, backCount=5, fmt='%(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()  
        sh.setFormatter(format_str) 

        th = logging.FileHandler(filename=filename, mode='a', encoding='utf-8', delay=False)
        th.setFormatter(format_str) 
        self.logger.addHandler(sh)  
        self.logger.addHandler(th)

