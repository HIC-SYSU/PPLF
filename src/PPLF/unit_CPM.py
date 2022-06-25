#!/usr/bin/python
# -*- coding: UTF-8 -*- 
# Author:   Hongwei Zhang
# FileName: unit_CPM.py
# Description:
import torch
import torch.nn as nn
import torch.nn.functional as F
from PPLF.unit_Transformer import Transformer
from PPLF.unit_CPM_config import *


config_vit_32_16 = get_config(image_size=256, feature_size=32, grid=16)


class CBR(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dilation=0):
        super(CBR,self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):

        y = self.relu(self.bn(self.conv(x)))

        return y

def upsampleLike(src,tar):

    src = F.interpolate(src,size=tar.shape[2:],mode='bilinear', align_corners=False)

    return src

class CPM1(nn.Module):
    def __init__(self, in_channels=3, mid_channels=12, out_channels=3):
        super(CPM1,self).__init__()

        self.cbrin = CBR(in_channels, out_channels)

        self.cbr1 = CBR(out_channels, mid_channels)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.cbr2 = CBR(mid_channels, mid_channels)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.cbr3 = CBR(mid_channels, mid_channels)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.cbr4 = CBR(mid_channels, mid_channels)
        # self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.transformer = Transformer(config=get_config(image_size=256, feature_size=32, grid=16), vis=False, in_channels=mid_channels)

        self.cbr4d = CBR(mid_channels * 2, mid_channels)

        self.cbr3d = CBR(mid_channels * 2, mid_channels)

        self.cbr2d = CBR(mid_channels * 2, mid_channels)

        self.cbr1d = CBR(mid_channels * 2, out_channels)


    def forward(self,x):

        xin = self.cbrin(x)

        x1 = self.cbr1(xin)
        x = self.pool1(x1)

        x2 = self.cbr2(x)
        x = self.pool2(x2)

        x3 = self.cbr3(x)
        x = self.pool3(x3)

        x4 = self.cbr4(x)

        xtr = self.transformer(x4)
        xtrup =  upsampleLike(xtr,x4)

        x4d = self.cbr4d(torch.cat((xtrup,x4),1))
        x4dup = upsampleLike(x4d,x3)

        x3d = self.cbr3d(torch.cat((x4dup,x3),1))
        x3dup = upsampleLike(x3d,x2)

        x2d = self.cbr2d(torch.cat((x3dup,x2),1))
        x2dup = upsampleLike(x2d,x1)

        x1d = self.cbr1d(torch.cat((x2dup,x1),1))

        y = x1d + xin

        return y


class CPM2(nn.Module):
    def __init__(self, in_channels=3, mid_channels=12, out_channels=3):
        super(CPM2,self).__init__()

        self.cbrin = CBR(in_channels, out_channels)

        self.cbr1 = CBR(out_channels, mid_channels)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.cbr2 = CBR(mid_channels, mid_channels)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.cbr3 = CBR(mid_channels, mid_channels)

        self.cbr4 = CBR(mid_channels, mid_channels)

        self.transformer = Transformer(config=get_config(image_size=256, feature_size=32, grid=16), vis=False, in_channels=mid_channels)

        self.cbr4d = CBR(mid_channels * 2, mid_channels)

        self.cbr3d = CBR(mid_channels * 2, mid_channels)

        self.cbr2d = CBR(mid_channels * 2, mid_channels)

        self.cbr1d = CBR(mid_channels * 2, out_channels)


    def forward(self,x):

        xin = self.cbrin(x)

        x1 = self.cbr1(xin)
        x = self.pool1(x1)

        x2 = self.cbr2(x)
        x = self.pool2(x2)

        x3 = self.cbr3(x)

        x4 = self.cbr4(x3)

        xtr = self.transformer(x4)
        xtrup =  upsampleLike(xtr,x4)

        x4d = self.cbr4d(torch.cat((xtrup,x4),1))

        x3d = self.cbr3d(torch.cat((x4d,x3),1))
        x3dup = upsampleLike(x3d,x2)

        x2d = self.cbr2d(torch.cat((x3dup,x2),1))
        x2dup = upsampleLike(x2d,x1)

        x1d = self.cbr1d(torch.cat((x2dup,x1),1))

        y = x1d + xin

        return y


class CPM3(nn.Module):
    def __init__(self, in_channels=3, mid_channels=12, out_channels=3):
        super(CPM3,self).__init__()

        self.cbrin = CBR(in_channels, out_channels)

        self.cbr1 = CBR(out_channels, mid_channels)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.cbr2 = CBR(mid_channels, mid_channels)

        self.cbr3 = CBR(mid_channels, mid_channels)

        self.cbr4 = CBR(mid_channels, mid_channels)

        self.transformer = Transformer(config=get_config(image_size=256, feature_size=32, grid=16), vis=False, in_channels=mid_channels)

        self.cbr4d = CBR(mid_channels * 2, mid_channels,)

        self.cbr3d = CBR(mid_channels * 2, mid_channels)

        self.cbr2d = CBR(mid_channels * 2, mid_channels)

        self.cbr1d = CBR(mid_channels * 2, out_channels)


    def forward(self,x):

        xin = self.cbrin(x)

        x1 = self.cbr1(xin)
        x = self.pool1(x1)

        x2 = self.cbr2(x)

        x3 = self.cbr3(x2)

        x4 = self.cbr4(x3)

        xtr = self.transformer(x4)
        xtrup =  upsampleLike(xtr,x4)

        x4d = self.cbr4d(torch.cat((xtrup,x4),1))

        x3d = self.cbr3d(torch.cat((x4d,x4d),1))

        x2d = self.cbr2d(torch.cat((x3d,x2),1))
        x2dup = upsampleLike(x2d,x1)

        x1d = self.cbr1d(torch.cat((x2dup,x1),1))

        y = x1d + xin

        return y


class CPM4(nn.Module):
    def __init__(self, in_channels=3, mid_channels=12, out_channels=3):
        super(CPM4,self).__init__()

        self.cbrin = CBR(in_channels,out_channels)

        self.cbr1 = CBR(out_channels,mid_channels)

        self.cbr2 = CBR(mid_channels,mid_channels)

        self.cbr3 = CBR(mid_channels,mid_channels)

        self.cbr4 = CBR(mid_channels,mid_channels)

        self.transformer = Transformer(config=get_config(image_size=256, feature_size=32, grid=16),vis=False, in_channels=mid_channels)

        self.cbr4d = CBR(mid_channels*2,mid_channels)

        self.cbr3d = CBR(mid_channels*2,mid_channels)

        self.cbr2d = CBR(mid_channels*2,mid_channels)

        self.cbr1d = CBR(mid_channels*2,out_channels)

    def forward(self,x):

        xin = self.cbrin(x)

        x1 = self.cbr1(xin)

        x2 = self.cbr2(x1)

        x3 = self.cbr3(x2)

        x4 = self.cbr4(x3)

        xtr = self.transformer(x4)
        xtrup =  upsampleLike(xtr,x4)

        x4d = self.cbr4d(torch.cat((xtrup,x4),1))

        x3d = self.cbr3d(torch.cat((x4d,x4d),1))

        x2d = self.cbr2d(torch.cat((x3d,x2),1))

        x1d = self.cbr1d(torch.cat((x2d,x1),1))

        out = x1d + xin

        return out

class CPM5(nn.Module):
    def __init__(self, in_channels=3, mid_channels=12, out_channels=3):
        super(CPM5,self).__init__()

        self.cbrin = CBR(in_channels,out_channels)

        self.cbr1 = CBR(out_channels,mid_channels)

        self.cbr2 = CBR(mid_channels,mid_channels)

        self.cbr3 = CBR(mid_channels,mid_channels)

        self.cbr4 = CBR(mid_channels,mid_channels)

        self.transformer = Transformer(config=get_config(image_size=256, feature_size=32, grid=16),vis=False, in_channels=mid_channels)

        self.cbr4d = CBR(mid_channels*2,mid_channels)

        self.cbr3d = CBR(mid_channels*2,mid_channels)

        self.cbr2d = CBR(mid_channels*2,mid_channels)

        self.cbr1d = CBR(mid_channels*2,out_channels)

    def forward(self,x):

        xin = self.cbrin(x)

        x1 = self.cbr1(xin)

        x2 = self.cbr2(x1)

        x3 = self.cbr3(x2)

        x4 = self.cbr4(x3)

        xtr = self.transformer(x4)

        xtrup =  upsampleLike(xtr,x4)

        x4d = self.cbr4d(torch.cat((xtrup,x4),1))

        x3d = self.cbr3d(torch.cat((x4d,x4d),1))

        x2d = self.cbr2d(torch.cat((x3d,x2),1))

        x1d = self.cbr1d(torch.cat((x2d,x1),1))

        out = x1d + xin

        return out

class Encoder(nn.Module):
    def __init__(self,in_channels=3):
        super(Encoder,self).__init__()

        self.stage1 = CPM1(in_channels,32,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = CPM2(64,32,128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = CPM3(128,64,256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = CPM4(256,128,512)

        self.stage5 = CPM5(512,256,512)

    def forward(self,x):

        x1 = self.stage1(x)
        x = self.pool12(x1)

        x2 = self.stage2(x)
        x = self.pool23(x2)

        x3 = self.stage3(x)
        x = self.pool34(x3)

        x4 = self.stage4(x)
        x = x4

        x5 = self.stage5(x)

        return x1, x2, x3, x4, x5
