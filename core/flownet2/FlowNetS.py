'''
Portions of this code copyright 2017, Clement Pinard
'''

import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from collections import OrderedDict

from .submodules import *
'Parameter count : 38,676,504 '

MAX_IMAGE = 65535.


class InputPadder:
    """ Pads images such that dimensions are divisible by 32 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 64) + 1) * 64 - self.ht) % 64
        pad_wd = (((self.wd // 64) + 1) * 64 - self.wd) % 64
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


class FlowNetS(nn.Module):
    def __init__(self, args, input_channels = 12, batchNorm=False, div_flow = 20):
        super(FlowNetS,self).__init__()

        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.conv1   = conv(self.batchNorm,  input_channels,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample5 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample6 = nn.Upsample(scale_factor=16, mode='bilinear')

        # loaded = torch.load('/data2/liuziyang/HDR/HDR_Ghost_Removal/RAFT-master/checkpoints/flownets-hdr-imf-short2long-fullloss-smooth0.01-newdata-occ.pth', map_location='cpu')
        # load_net_clean = OrderedDict()
        # for k, v in loaded.items():
        #     if 'module.flownets_1.' in k:  # and 'transform.' not in k:
        #         k = k.replace('module.flownets_1.', '')
        #         load_net_clean[k] = v
        # self.load_state_dict(load_net_clean, strict=True)

    def forward(self, x, training=True, padder=None):
        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        
        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        
        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        # if self.training:
        #     return flow2,flow3,flow4,flow5,flow6
        # else:
        #     return flow2,

        if training:
            # return flow2,flow3,flow4,flow5,flow6
            flow2 = self.upsample1(flow2 * self.div_flow)
            flow3 = self.upsample1(self.upsample3(flow3 * 2) * self.div_flow)
            flow4 = self.upsample1(self.upsample4(flow4 * 4) * self.div_flow)
            flow5 = self.upsample1(self.upsample5(flow5 * 8) * self.div_flow)
            flow6 = self.upsample1(self.upsample6(flow6 * 16) * self.div_flow)
            return [padder.unpad(flow6[:, [1,0], ...]), padder.unpad(flow5[:, [1,0], ...]), padder.unpad(flow4[:, [1,0], ...]),
                    padder.unpad(flow3[:, [1,0], ...]), padder.unpad(flow2[:, [1,0], ...])]
        else:
            return flow2

