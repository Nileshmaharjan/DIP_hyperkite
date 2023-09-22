from math import ceil
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.functional as F
import math


# Updated PAN prediction network...
class Predict_PAN(nn.Module):
    def __init__(self, spectral_bands):
        super(Predict_PAN, self).__init__()
        r = 2
        self.AvgPool    = nn.AdaptiveAvgPool2d(1)
        self.conv1      = nn.Conv2d(in_channels=spectral_bands, out_channels=int(spectral_bands/r), kernel_size=1)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = nn.Conv2d(in_channels=int(spectral_bands/r), out_channels=spectral_bands, kernel_size=1)
        self.SoftMax    = nn.Softmax(dim=1)


    def forward(self, net_output, mode="GAP"):
        if mode=="GAP":
            R           = self.SoftMax(self.conv2(self.relu(self.conv1(self.AvgPool(net_output)))))
            PAN_pred    = torch.sum(net_output*R.expand_as(net_output), dim=1)
            return PAN_pred
        elif mode=="MEAN":
            PAN_pred    = torch.mean(net_output, dim=1)
            return PAN_pred


# import torch
# import torch.nn as nn
#
# class SpatialAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SpatialAttention, self).__init__()
#         self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         attention = self.sigmoid(self.conv(x))
#         return attention
#
# class Predict_PAN(nn.Module):
#     def __init__(self, spectral_bands):
#         super(Predict_PAN, self).__init__()
#         reduction_factor = 2
#         self.pool = nn.MaxPool2d(kernel_size=2)
#         self.conv1 = nn.Conv2d(in_channels=spectral_bands, out_channels=int(spectral_bands/reduction_factor), kernel_size=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(in_channels=int(spectral_bands/reduction_factor), out_channels=int(spectral_bands/reduction_factor), kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=int(spectral_bands/reduction_factor), out_channels=spectral_bands, kernel_size=1)
#         self.attention = SpatialAttention(int(spectral_bands/reduction_factor))
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, net_output,  mode="GAP"):
#         x = self.relu(self.conv1(net_output))
#         x = self.pool(x)
#         attention = self.attention(x)  # Spatial attention
#         x = attention * x  # Element-wise multiplication
#         x = self.relu(self.conv2(x))
#         # Resize x to match the size of net_output
#         x = nn.functional.interpolate(x, size=net_output.size()[2:], mode='bilinear', align_corners=False)
#         x = self.conv3(x)
#         R = self.softmax(x)
#         PAN_pred = torch.sum(net_output * R.expand_as(net_output), dim=1)
#         return PAN_pred


