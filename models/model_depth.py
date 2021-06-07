from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch.nn as nn

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class DepthNet(nn.Module):

    def __init__(self):
        super(DepthNet, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2    2 layers

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4   2 layers
        num_stages = 3
        blocks = BasicBlock
        num_blocks = [4, 4, 4]
        num_channels = [32, 32, 128]
        self.stage = self._make_stages(num_stages, blocks, num_blocks, num_channels)
        self.transition1 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.transition2 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

    def _make_one_stage(self, stage_index, block, num_blocks, num_channels):
        layers = []
        for i in range(0, num_blocks[stage_index]):
            layers.append(
                block(
                    num_channels[stage_index],
                    num_channels[stage_index]
                )
            )
        return nn.Sequential(*layers)

    def _make_stages(self, num_stages, block, num_blocks, num_channels):
        branches = []

        for i in range(num_stages):
            branches.append(
                self._make_one_stage(i, block, num_blocks, num_channels)
            )
        return nn.ModuleList(branches)

    def forward(self, d):
        #depth branch
        d = self.relu1_1(self.bn1_1(self.conv1_1(d)))
        d = self.relu1_2(self.bn1_2(self.conv1_2(d)))
        d0 = self.pool1(d)  # (128x128)*64

        d = self.relu2_1(self.bn2_1(self.conv2_1(d0)))
        d = self.relu2_2(self.bn2_2(self.conv2_2(d)))
        d1 = self.pool2(d)  # (64x64)*128
        dt2 = self.transition1(d1)
        d2 = self.stage[0](dt2)
        d3 = self.stage[1](d2)
        dt4 = self.transition2(d3)
        d4 = self.stage[2](dt4)
        return d0, d1, d2, d3, d4

    def init_weights(self):
        logger.info('=> Depth model init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)