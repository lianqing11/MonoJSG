import torch
from torch import nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models import NECKS

@NECKS.register_module()
class IdentityNeck(nn.Module):
    def __init__(self):
        super().__init__()
    @auto_fp16()
    def forward(self, x):
        return x

    def init_weights(self):
        pass



class BasicBlock2d(nn.Module):
    """ 3x3x3 Resnet Basic Block"""
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm='BN', drop=0):
        super(BasicBlock2d, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, 1, dilation)
        self.bn1 = get_norm_2d(norm, planes)
        # self.drop1 = nn.Dropout(drop, True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1, 1, dilation)
        self.bn2 = get_norm_2d(norm, planes)
        #self.drop2 = nn.Dropout(drop, True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.drop1(out) # drop after both??
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.drop2(out) # drop after both??

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock3dV2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock3dV2, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.BatchNorm3d(out_channels)
        if self.stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.stride != 1:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out