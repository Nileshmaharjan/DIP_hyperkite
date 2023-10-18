import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.functional as F
import sys
import os
import numpy as np
import cv2

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

RELUSLOPE = 0.2





class kitenetwithsk(nn.Module):
    def __init__(self, config):
        super(kitenetwithsk, self).__init__()
        self.is_DHP_MS = config["is_DHP_MS"]
        self.in_channels = config[config["train_dataset"]]["spectral_bands"] + 1
        self.out_channels = config[config["train_dataset"]]["spectral_bands"]

        filters = [32, 64, 128]

        # ENCODER FILTERS
        self.encoder1 = nn.Conv2d(self.in_channels, filters[0], 3, stride=1, padding=1)
        self.ebn1 = nn.BatchNorm2d(filters[0])
        self.encoder2 = nn.Conv2d(filters[0], filters[1], 3, stride=1, padding=1)
        self.ebn2 = nn.BatchNorm2d(filters[1])
        self.encoder3 = nn.Conv2d(filters[1], filters[2], 3, stride=1, padding=1)
        self.ebn3 = nn.BatchNorm2d(filters[2])

        # BOTTELENECK FILTERS
        self.endec_conv = nn.Conv2d(filters[2], filters[2], 3, stride=1, padding=1)
        self.endec_bn = nn.BatchNorm2d(filters[2])

        # DECODER FILTERS
        self.decoder1 = nn.Conv2d(2 * filters[2], filters[1], 3, stride=1, padding=1)  # b, 1, 28, 28
        self.dbn1 = nn.BatchNorm2d(filters[1])
        self.decoder2 = nn.Conv2d(2 * filters[1], filters[0], 3, stride=1, padding=1)
        self.dbn2 = nn.BatchNorm2d(filters[0])
        self.decoder3 = nn.Conv2d(2 * filters[0], self.out_channels, 3, stride=1, padding=1)
        self.dbn3 = nn.BatchNorm2d(self.out_channels)

        # FINAL CONV LAYER
        self.final_conv = nn.Conv2d(self.out_channels, self.out_channels, 1)

        # RELU
        self.relu = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)

    def forward(self, X_MS, X_PAN):
        if not self.is_DHP_MS:
            X_MS_UP = F.interpolate(X_MS, scale_factor=(4, 4), mode='bilinear')
        else:
            X_MS_UP = X_MS

        x = torch.cat((X_MS_UP, X_PAN.unsqueeze(1)), dim=1)

        # ENCODER
        out = self.relu(self.ebn1(self.encoder1(x)))
        t1 = out

        out = F.interpolate(out, scale_factor=(2, 2), mode='bilinear')
        out = self.relu(self.ebn2(self.encoder2(out)))
        t2 = out

        out = F.interpolate(out, scale_factor=(2, 2), mode='bilinear')
        out = self.relu(self.ebn3(self.encoder3(out)))
        t3 = out

        # BOTTLENECK
        out = F.interpolate(out, scale_factor=(2, 2), mode='bilinear')
        out = self.relu(self.endec_bn(self.endec_conv(out)))

        # DECODER
        out = F.max_pool2d(out, 2, 2)
        out = torch.cat((out, t3), dim=1)
        out = self.relu(self.dbn1(self.decoder1(out)))

        out = F.max_pool2d(out, 2, 2)
        out = torch.cat((out, t2), dim=1)
        out = self.relu(self.dbn2(self.decoder2(out)))

        out = F.max_pool2d(out, 2, 2)
        out = torch.cat((out, t1), dim=1)
        out = self.relu(self.dbn3(self.decoder3(out)))

        # OUTPUT CONV
        out = self.final_conv(out)

        # FINAL OUTPUT
        out = out + X_MS_UP

        output = {"pred": out}
        print('output')
        return output

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class attentionkitenet(nn.Module):
    def __init__(self, config):
        super(attentionkitenet, self).__init__()
        self.is_DHP_MS = config["is_DHP_MS"]
        self.in_channels = config[config["train_dataset"]]["spectral_bands"] + 1
        self.out_channels = config[config["train_dataset"]]["spectral_bands"]
        self.N_Filters = 64
        self.N_modules = config["N_modules"]

        self.encoder1 = nn.Conv2d(self.in_channels, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.decoder1 = nn.Conv2d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder2 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(64, self.out_channels, 3, stride=1, padding=1)

        self.relu = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)

        self.CBAM = CBAM(128)  # Use CBAM for the encoder3 output

    def forward(self, X_MS, X_PAN):
        if not self.is_DHP_MS:
            X_MS_UP = F.interpolate(X_MS, scale_factor=(4, 4), mode='bilinear')
        else:
            X_MS_UP = X_MS

        x = torch.cat((X_MS_UP, X_PAN.unsqueeze(1)), dim=1)

        out = self.relu(F.interpolate(self.encoder1(x), scale_factor=(2, 2), mode='bilinear'))
        t1 = out
        out = self.relu(F.interpolate(self.encoder2(out), scale_factor=(2, 2), mode='bilinear'))
        t2 = out
        out = self.relu(F.interpolate(self.encoder3(out), scale_factor=(2, 2), mode='bilinear'))
        out = self.CBAM(out)  # Apply CBAM to the encoder3 output

        out = self.relu(F.max_pool2d(self.decoder1(out), 2, 2))
        out = torch.cat((t2, out), dim=1)

        out = self.relu(F.max_pool2d(self.decoder2(out), 2, 2))
        out = torch.cat((t1, out), dim=1)

        out = self.relu(F.max_pool2d(self.decoder3(out), 2, 2))

        out = out + X_MS_UP
        output = {"pred": out}

        return output
