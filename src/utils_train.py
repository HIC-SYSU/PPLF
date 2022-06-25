#!/usr/bin/python
# -*- coding: UTF-8 -*- 
# Author:   Hongwei Zhang
# FileName: utils_train.py
# Description:

import os
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from utils import *
from data_helper import DSA_keyframe
from index import all_index

def main(fold_th, args, build_model_and_optimzer):
    setup_seed(888)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
    global dice_best
    dice_best = 0.9

    global save_path
    save_path = 'train_result/' + args.info_help +'_'+ str(int(time.time())) + '/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    global log
    log = Logger(filename=save_path + 'log.txt', level='info')

    global writer
    writer = SummaryWriter(log_dir=save_path + 'runs')

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = args.data_path + args.data_size + args.data_vessel

    train_data = DSA_keyframe(data_path, is_train_val_test='1', online_aug=args.aug,offline_aug=args.aug,folds_all=args.folds_all,fold_th=fold_th)
    n_train = len(train_data)
    n_train_true = train_data.train_true_len
    val_data = DSA_keyframe(data_path, is_train_val_test='2', online_aug=False,offline_aug=False,folds_all=args.folds_all,fold_th=fold_th)
    n_val = len(val_data)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    model, calAll, optimizer, lr_scheduler = build_model_and_optimzer(args)

    logging_information(log, args, save_path, n_train_true, n_train, n_val)
    log.logger.info(f'Epoch\tLoss\tDice\tPrec\tRecall\tJAccard\tHD')

    for epoch in range(args.epochs):
        epoch_log = epoch + 1

        train(train_loader, val_loader, model, calAll, optimizer, epoch_log, args)
        eval(val_loader, model, calAll, epoch_log, args)

        lr_scheduler.step()
        writer.add_scalars('learning_rate_e', {'': optimizer.param_groups[0]['lr']}, epoch_log)

def train(train_loader,val_loader, model, calAll, optimizer, epoch_log,args):
    loss_meter_list = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    dice_meter = AverageMeter()
    Jaccard_meter = AverageMeter()
    HD_meter = AverageMeter()

    model.train()
    for step, batch in enumerate(train_loader):
        imgs = batch['patient'].to(device=device, dtype=torch.float32)
        gt = batch['seg_label'].to(device=device, dtype=torch.float32)
        imgs_name = batch['patient_name']

        pred, loss_list, loss_name_list= calAll(gt, model(imgs))
        assert len(loss_list) ==  len(loss_name_list)

        optimizer.zero_grad()
        loss_list[0].backward()
        nn.utils.clip_grad_value_(model.parameters(), 0.1)
        optimizer.step()

        precision, recall, Dice, Jaccard, HD = all_index(gt, pred)

        n = imgs.size(0)
        for i in range(len(loss_list)):
            loss_meter_list[i].update(loss_list[i].item(),n)
        precision_meter.update(precision.item(), n)
        recall_meter.update(recall.item(), n)
        dice_meter.update(Dice.item(), n)
        Jaccard_meter.update(Jaccard.item(), n)
        HD_meter.update(HD.item(), n)

        current_iter = (epoch_log-1) * len(train_loader) + step + 1

        if current_iter == 1 or current_iter % 100 == 0:
            print(save_path+'\ntrain:epoch=', epoch_log, "batch=", step+1, '\nloss=', '%.4f'%loss_list[0].item(),
                  'dice=', '%.4f'%Dice.item(), 'Jaccard=', '%.4f'%Jaccard.item(), 'HD=', '%.4f'%HD.item(),
                  'precision=', '%.4f'%precision.item(), 'recall=', '%.4f'%recall.item())

            for i in range(len(loss_list)):
                writer.add_scalars('loss_b', {'train_'+loss_name_list[i]: loss_list[i].item()}, current_iter)
            writer.add_scalars('dice_b', {'train': Dice.item()}, current_iter)
            writer.add_scalars('Jaccard_b', {'train': Jaccard.item()}, current_iter)
            writer.add_scalars('HD_b', {'train': HD.item()}, current_iter)
            writer.add_scalars('precision_b', {'train': precision.item()}, current_iter)
            writer.add_scalars('recall_b', {'train': recall.item()}, current_iter)

        if args.save_pic_train == True:
            if epoch_log == args.epochs:
                pred = (pred > 0.5).float()
                pic_save_path = save_path + 'train_pic_save/train_Epoch' + str(epoch_log)+'/'
                if not os.path.exists(pic_save_path): os.makedirs(pic_save_path)
                color_predict_save(imgs, gt, pred, imgs_name, save_path=pic_save_path + "%d_" % step)

    for i in range(len(loss_list)):
        writer.add_scalars('loss_e', {'train_'+loss_name_list[i]: loss_meter_list[i].avg}, epoch_log)
    writer.add_scalars('dice_e', {'train': dice_meter.avg}, epoch_log)
    writer.add_scalars('precision_e', {'train': precision_meter.avg}, epoch_log)
    writer.add_scalars('recall_e', {'train': recall_meter.avg}, epoch_log)

def eval(val_loader, model, calAll, epoch_log, args):
    loss_meter_list = [AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()]
    dice_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    Jaccard_meter = AverageMeter()
    HD_meter = AverageMeter()

    val_save_path = save_path+'val_txt_save/'
    if not os.path.exists(val_save_path): os.makedirs(val_save_path)
    frlog = open(val_save_path + 'epoch' + str(epoch_log) + '.txt', 'w')
    print('Number\tPic_name\tDice\tPrec\tRecall\tJAccard\tHD', file=frlog)

    model.eval()

    for step,batch in enumerate(val_loader):
        imgs = batch['patient'].to(device=device, dtype=torch.float32)
        gt = batch['seg_label'].to(device=device, dtype=torch.float32)
        imgs_name = batch['patient_name']

        with torch.no_grad():
            pred, loss_list, loss_name_list = calAll(gt, model(imgs))

        pred = (pred > 0.5).float()

        precision, recall, Dice, Jaccard, HD = all_index(gt, pred)

        n = imgs.size(0)
        for i in range(len(loss_list)):
            loss_meter_list[i].update(loss_list[i].item(),n)
        precision_meter.update(precision.item(), n)
        recall_meter.update(recall.item(), n)
        dice_meter.update(Dice.item(), n)
        Jaccard_meter.update(Jaccard.item(), n)
        HD_meter.update(HD.item(), n)

        print("%d" % step +'\t'+ imgs_name[0] +'\t'+ str('%.4f'%dice_meter.val) +'\t'+ str('%.4f'%precision_meter.val) +'\t'+ str('%.4f'%recall_meter.val)
              + '\t' + str('%.4f' % Jaccard_meter.val)+ '\t' + str('%.4f' % HD_meter.val)
              , file=frlog)

        if args.save_pic_val == True:
            pic_save_path = save_path + 'val_pic_save/val_Epoch' + str(epoch_log) + '/'
            if not os.path.exists(pic_save_path): os.makedirs(pic_save_path)
            imgs_name[0] = imgs_name[0][0:-4] + '_' + str('%.2f' % (dice_meter.val * 100)) + '.jpg'
            color_predict_save(imgs, gt, pred, imgs_name, save_path=pic_save_path + "%d_" % step)

    print('average:\t' + str('%.4f'%dice_meter.avg) + '\t' + str('%.4f'%precision_meter.avg) + '\t' + str('%.4f'%recall_meter.avg)
          + '\t' + str('%.4f' % Jaccard_meter.avg)
          + '\t' + str('%.4f' % HD_meter.avg)
          , file=frlog)
    frlog.close()

    for i in range(len(loss_list)):
        writer.add_scalars('loss_e', {'val_'+loss_name_list[i]: loss_meter_list[i].avg}, epoch_log)
    writer.add_scalars('dice_e', {'val': dice_meter.avg}, epoch_log)
    writer.add_scalars('precision_e', {'val': precision_meter.avg}, epoch_log)
    writer.add_scalars('recall_e', {'val': recall_meter.avg}, epoch_log)

    log.logger.info(str(epoch_log) + '\t' + str('%.4f' % loss_meter_list[0].avg) + '\t' + str('%.4f' % dice_meter.avg) + '\t'
                    + str('%.4f' % precision_meter.avg) + '\t' + str('%.4f' % recall_meter.avg) + '\t'
                    + str('%.4f' % Jaccard_meter.avg) + '\t' + str('%.4f' % HD_meter.avg) + '\t')


    val_model_save = args.save_model
    if val_model_save == True:
        model_save_path = save_path + 'model_save/'
        if not os.path.exists(model_save_path): os.makedirs(model_save_path)
        torch.save(model.state_dict(), model_save_path + f'CP_epoch{epoch_log}.pth')

    return dice_meter.avg

