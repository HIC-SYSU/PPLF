#!/usr/bin/python
# -*- coding: UTF-8 -*- 
# Author:   Hongwei Zhang
# FileName: index.py
# Description:
import torch

def cdist(x, y):
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances

def all_index(gt, map):
    mask = torch.gt(map, 0.5).cuda()
    mask = mask.type(torch.float32).cuda()

    gtCnt = torch.sum(gt).cuda()
    hitMap = torch.where(gt > 0, mask, torch.zeros(mask.size()).cuda()).cuda()

    hitCnt = torch.sum(hitMap)
    algCnt = torch.sum(mask)

    precison = hitCnt / (algCnt + 1e-12)
    recall = hitCnt / (gtCnt + 1e-12)

    DSC = 2 * hitCnt /(gtCnt + algCnt + 1e-32)

    Jaccard = hitCnt / (gtCnt + algCnt - hitCnt +  1e-32)

    d2_matrix = cdist(gt, map)
    term_1 = torch.mean(torch.min(d2_matrix, 1)[0])
    term_2 = torch.mean(torch.min(d2_matrix, 0)[0])
    HD = term_1 + term_2

    return precison, recall, DSC, Jaccard, HD

