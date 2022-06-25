#!/usr/bin/python
# -*- coding: UTF-8 -*- 
# Author:   Hongwei Zhang
# FileName: unit_IPM_BPM.py
# Description:
import torch
import torch.nn as nn
import torch.nn.functional as F

class reduceDim(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(reduceDim, self).__init__()
        self.r0 = nn.Sequential(
            nn.Conv2d(in_dim[0], out_dim[0], kernel_size=1),nn.ReLU()
        )
        self.r1 = nn.Sequential(
            nn.Conv2d(in_dim[1], out_dim[1], kernel_size=1),nn.ReLU()
        )
        self.r2 = nn.Sequential(
            nn.Conv2d(in_dim[2], out_dim[2], kernel_size=1),nn.ReLU()
        )
        self.r3 = nn.Sequential(
            nn.Conv2d(in_dim[3], out_dim[3], kernel_size=1),nn.ReLU()
        )
        self.r4 = nn.Sequential(
            nn.Conv2d(in_dim[4], out_dim[4], kernel_size=1),nn.ReLU()
        )

    def forward(self, x1, x2, x3, x4, x5):
        y1 = self.r0(x1)
        y2 = self.r1(x2)
        y3 = self.r2(x3)
        y4 = self.r3(x4)
        y5 = self.r4(x5)

        return y1,y2,y3,y4,y5

def upsampleLike(src,tar):

    src = F.interpolate(src,size=tar.shape[2:],mode='bilinear', align_corners=False)

    return src

def side_upsample(x):
    n = len(x)
    upsample_list = []
    upsample_list.append(x[0])
    for i in range(1,n):
        x_upsample = upsampleLike(x[i], x[0])
        upsample_list.append(x_upsample)
    return upsample_list

class IB(nn.Module):
    def __init__(self, in_channels):
        super(IB, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = int(in_channels / 4)

        self.branch11 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_channels), nn.ReLU())
        self.branch12 = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.mid_channels), nn.ReLU())
        self.branch13 = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.mid_channels), nn.ReLU())

        self.branch21 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_channels), nn.ReLU())
        self.branch22 = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.mid_channels), nn.ReLU())
        self.branch23 = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.mid_channels), nn.ReLU())

        self.branch31 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_channels), nn.ReLU())
        self.branch32 = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.mid_channels), nn.ReLU())
        self.branch33 = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.mid_channels), nn.ReLU())

        self.branch41 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_channels), nn.ReLU())
        self.branch42 = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.mid_channels), nn.ReLU())
        self.branch43 = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.mid_channels), nn.ReLU())

        self.fusion = nn.Sequential(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(self.in_channels), nn.ReLU())

    def forward(self, x):
        b11 = self.branch11(x)
        b12 = self.branch12(b11)
        b13 = self.branch13(b12)

        b21 = self.branch21(x) + b13
        b22 = self.branch22(b21)
        b23 = self.branch23(b22)

        b31 = self.branch31(x) + b23
        b32 = self.branch32(b31)
        b33 = self.branch33(b32)

        b41 = self.branch41(x) + b33
        b42 = self.branch42(b41)
        b43 = self.branch43(b42)

        y = self.fusion(torch.cat((b13, b23, b33, b43), 1))

        return y

class IPM(nn.Module):
    def __init__(self, in_channels1, in_channels2, isUpsampling=True):
        super(IPM, self).__init__()
        self.in_channels1 = in_channels1
        self.in_channels2 = in_channels2
        self.isUpsampling = isUpsampling

        self.up = nn.Sequential(nn.Conv2d(self.in_channels2, self.in_channels1, 7, 1, 3),
                                nn.BatchNorm2d(self.in_channels1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))

        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())
        self.output_map = nn.Conv2d(self.in_channels1, 1, 7, 1, 3)

        self.ib1 = IB(self.in_channels1)
        self.ib2 = IB(self.in_channels1)
        self.lambda1 = nn.Parameter(torch.ones(1))
        self.lambda2 = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm2d(self.in_channels1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.in_channels1)
        self.relu2 = nn.ReLU()

    def forward(self, feature_c, feature_h, y, y_rev):

        if self.isUpsampling == True:
            feature_h = self.up(feature_h)
            y = self.input_map(y)
            y_rev = self.input_map(y_rev)

        feature_f = feature_c * y
        feature_b = feature_c * y_rev

        feature_fi = self.ib1(feature_f)
        feature_bi = self.ib2(feature_b)

        feature_ip1 = feature_h - (self.lambda2 * feature_bi)
        feature_ip1 = self.bn1(feature_ip1)
        feature_ip1 = self.relu1(feature_ip1)

        feature_ip2 = feature_ip1 + (self.lambda1 * feature_fi)
        feature_ip2 = self.bn2(feature_ip2)
        feature_ip2 = self.relu2(feature_ip2)

        return feature_ip2

class BPM(nn.Module):
    def __init__(self, in_channels, mid_channels, isUpsampling = True):
        super(BPM, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.isUpsampling = isUpsampling

        self.conv_bg = nn.Sequential(
            nn.Conv2d(self.in_channels+1, self.mid_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(self.mid_channels, 1, kernel_size=1), nn.ReLU()
        )

        self.conv_fg = nn.Sequential(
            nn.Conv2d(self.in_channels+1, self.mid_channels, kernel_size=3, padding=1),nn.ReLU(),
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, padding=1),nn.ReLU()
        )

        self.out = nn.Conv2d(self.in_channels, 1, 3, padding=1)

    def forward(self, feature_ip, y):
        y = upsampleLike(y,  feature_ip)
        feature = torch.cat((feature_ip, y), dim=1)

        feature_bp = self.conv_fg(feature)

        y_sigmoid = torch.sigmoid(y)

        y_rev = self.conv_bg(feature)
        y_rev_sigmoid = torch.sigmoid(y_rev)

        edge = torch.mul(y_rev_sigmoid, y_sigmoid)
        feature_edge = torch.mul(feature_bp,edge)

        feature_bp = feature_bp + feature_edge
        y_bp = self.out(feature_edge)
        y_bp = y + y_bp

        return feature_bp, y_bp, y_rev

class Block(nn.Module):
    def __init__(self, channel1, channel2, isUpsampling = True):
        super(Block, self).__init__()
        self.ipm = IPM(channel1, channel2, isUpsampling = isUpsampling)
        self.bpm = BPM(channel1, channel2, isUpsampling = isUpsampling)

    def forward(self, feature_c, feature_h, y, y_rev):
        y_refine = self.ipm(feature_c, feature_h, y, y_rev)
        feature_out, y, y_rev = self.bpm(y_refine, y)
        return feature_out, y, y_rev

class Decoder(nn.Module):
    def __init__(self, in_dim):
        super(Decoder, self).__init__()
        self.conv = nn.Conv2d(in_dim[4],1,3,padding=1)

        self.block4 = Block(in_dim[3], in_dim[3], isUpsampling = False)
        self.block3 = Block(in_dim[2], in_dim[2])
        self.block2 = Block(in_dim[1], in_dim[1])
        self.block1 = Block(in_dim[0], in_dim[0])

    def forward(self, x1, x2, x3, x4, x5):
        y5 = self.conv(x5)
        y5_rev = 1 - y5
        feature4, y4, y4_rev = self.block4(x4, x5, y5, y5_rev)
        feature3, y3, y3_rev = self.block3(x3, feature4, y4, y4_rev)
        feature2, y2, y2_rev = self.block2(x2, feature3, y3, y3_rev)
        feature1, y1, y1_rev = self.block1(x1, feature2, y2, y2_rev)

        return [y1, y2, y3, y4, y5], [y1_rev, y2_rev, y3_rev, y4_rev, y5_rev]


