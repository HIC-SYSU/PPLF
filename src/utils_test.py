#!/usr/bin/python
# -*- coding: UTF-8 -*- 
# Author:   Hongwei Zhang
# FileName: util_test.py
# Description:
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from utils import *
from data_helper import DSA_keyframe
from index import all_index

def main(fold_th, args, build_model_and_optimzer):
    setup_seed(888)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num

    global save_path
    save_path = 'train_result/' + args.model_path +'/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, calAll, optimizer, lr_scheduler = build_model_and_optimzer(args)

    data_path = args.data_path + args.data_size + args.data_vessel

    test_data = DSA_keyframe(data_path, is_train_val_test='3', online_aug=False,offline_aug=False,folds_all=args.folds_all,fold_th=fold_th)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    eval(test_loader, model, calAll, args.test_epoch, args)

def eval(test_loader, model, calAll, epoch_log, args):
    loss_meter_list = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
    dice_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    Jaccard_meter = AverageMeter()
    HD_meter = AverageMeter()

    test_save_path = save_path+'test_txt_save/'
    if not os.path.exists(test_save_path): os.makedirs(test_save_path)
    frlog = open(test_save_path + 'epoch' + str(epoch_log) + '.txt', 'w')
    print('Number\tPic_name\tDice\tPrec\tRecall\tJAccard\tHD', file=frlog)

    model.eval()

    model_path = './train_result/' + args.model_path + '/val_model_save/CP_epoch'+ str(epoch_log) +'.pth'
    model.load_state_dict(torch.load(model_path))

    for step,batch in enumerate(test_loader):
        imgs = batch['patient'].to(device=device, dtype=torch.float32)
        gt = batch['seg_label'].to(device=device, dtype=torch.float32)
        imgs_name = batch['patient_name']

        with torch.no_grad():
            pred, loss_list, loss_name_list = calAll(gt, model(imgs))

        pred = (pred > 0.5).float()

        precision, recall, F_score, Jaccard, HD = all_index(gt, pred)

        n = imgs.size(0)
        for i in range(len(loss_list)):
            loss_meter_list[i].update(loss_list[i].item(),n)
        precision_meter.update(precision.item(), n)
        recall_meter.update(recall.item(), n)
        dice_meter.update(F_score.item(), n)
        Jaccard_meter.update(Jaccard.item(), n)
        HD_meter.update(HD.item(), n)


        print("%d" % step +'\t'+ imgs_name[0] +'\t'+ str('%.4f'%dice_meter.val) +'\t'+ str('%.4f'%precision_meter.val) +'\t'+ str('%.4f'%recall_meter.val)
              + '\t' + str('%.4f' % Jaccard_meter.val) + '\t' + str('%.4f' % HD_meter.val)
              , file=frlog)

        test_pic_save = args.save_pic_test
        if test_pic_save == True:
            pic_save_path = save_path + 'test_pic_save/test_Epoch'+ str(epoch_log) +'/'
            if not os.path.exists(pic_save_path): os.makedirs(pic_save_path)
            imgs_name[0] = imgs_name[0][0:-4] + '_' + str('%.2f' % (dice_meter.val * 100)) + '.bmp'
            color_predict_save(imgs, gt, pred, imgs_name, save_path=pic_save_path + "%d_" % step)

    print('average:\t' + str('%.4f'%dice_meter.avg) + '\t' + str('%.4f'%precision_meter.avg) + '\t' + str('%.4f'%recall_meter.avg)
          + '\t' + str('%.4f' % Jaccard_meter.avg)
          + '\t' + str('%.4f' % HD_meter.avg)
          , file=frlog)
    frlog.close()
