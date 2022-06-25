#!/usr/bin/python
# -*- coding: UTF-8 -*- 
# Author:   Hongwei Zhang
# FileName: PPLF__model.py
# Description:

from PPLF.unit_CPM import Encoder
from PPLF.unit_IPM_BPM import *

class PPLF_5side_5revside(nn.Module):
    def __init__(self, in_channels=3):
        super(PPLF_5side_5revside,self).__init__()

        self.encoder = Encoder(in_channels=in_channels)

        self.reducedim = reduceDim(in_dim=[64, 128, 256, 512, 512],out_dim = [32, 32, 32, 32, 32])

        self.decoder = Decoder(in_dim = [32, 32, 32, 32, 32])

    def forward(self, x):
        y1, y2, y3, y4, y5 = self.encoder(x)

        y1, y2, y3, y4, y5 = self.reducedim(y1, y2, y3, y4, y5)

        d_list, d_rev_list = self.decoder(y1, y2, y3, y4, y5)

        p_list = side_upsample(d_list)

        p_rev_list = side_upsample(d_rev_list)

        return p_list, p_rev_list

if __name__ == "__main__":
    # test
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    x = torch.randn((1, 3, 256, 256))
    net = PPLF_5side_5revside()
    y, y_rev = net(x,)
