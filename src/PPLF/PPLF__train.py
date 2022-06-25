#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Author:   Hongwei Zhang
# FileName: PPLF__train.py
# Description:

import sys
sys.path.append('../')
from utils_train import main
from utils import *
from PPLF.PPLF__model import PPLF_5side_5revside
import torch.nn as nn

def get_args_final():
    parser = get_args()
    parser.add_argument('--optim', dest='optim', type=str, default='Adam',
                        help='optimizer: Sgd, Adam, RMSprop')
    parser.add_argument('--weight_decay', dest='weight_decay', default=1e-5, type=float,
                        help='weight decay (default: 1e-5)')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--step_size', dest='step_size', type=int, default=2,
                        help='step_size')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.5,
                        help='gamma')
    parser.add_argument('--model_name', dest='model_name', type=str, default='PPLF',
                        help='model name')
    args = parser.parse_args()

    return args

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = PPLF_5side_5revside(in_channels=3)
    def forward(self, x):
        [y, pre1, pre2, pre3, pre4], [y_rev1, pre_rev2, pre_rev3, pre_rev4, pre_rev5] = self.model(x)

        return y, pre1, pre2, pre3, pre4, y_rev1, pre_rev2, pre_rev3, pre_rev4, pre_rev5

class CalAll(nn.Module):
    def __init__(self, args=None):
        super(CalAll, self).__init__()
        self.args = args
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, gt, model_out):
        y, pre1, pre2, pre3, pre4, y_rev1, pre_rev2, pre_rev3, pre_rev4, pre_rev5 =\
            model_out[0],model_out[1],model_out[2],model_out[3],model_out[4],\
            model_out[5],model_out[6],model_out[7],model_out[8],model_out[9]

        y_sigmoid = torch.sigmoid(y)

        BCE_loss = self.bce_loss(y, gt)
        loss01 = self.bce_loss(pre1, gt)
        loss02 = self.bce_loss(pre2, gt)
        loss03 = self.bce_loss(pre3, gt)
        loss04 = self.bce_loss(pre4, gt)


        loss11 = self.bce_loss(y_rev1, 1-gt)
        loss12 = self.bce_loss(pre_rev2, 1-gt)
        loss13 = self.bce_loss(pre_rev3, 1-gt)
        loss14 = self.bce_loss(pre_rev4, 1-gt)
        loss15 = self.bce_loss(pre_rev5, 1-gt)

        SIDE_loss = loss01 + loss02 + loss03 + loss04 + loss11+loss12+loss13+loss14+loss15

        loss_total = BCE_loss + SIDE_loss
        loss_list = [loss_total, BCE_loss, SIDE_loss, ]
        loss_name_list = ['total','main_loss', 'SIDE_loss', ]

        return y_sigmoid, loss_list, loss_name_list

def build_model_and_optimzer(args):
    model = Model()
    model.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if len(args.gpu_num) > 2:
        model = nn.DataParallel(model)

    calAll = CalAll(args)

    optimizer = build_optimizer(optim=args.optim, model=model, lr=args.base_lr, weight_decay=args.weight_decay, momentum=args.momentum)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return model, calAll, optimizer, lr_scheduler

if __name__ == '__main__':
    args =  get_args_final()
    main(args.fold_th, args, build_model_and_optimzer)


