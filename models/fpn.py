"""
Feature Pyramid Network for Object Detection

"""

import torch
import torch.nn.functional as F
from torchvision.models import resnet
import torch.nn as nn
from .resnet import ResNet50Stages


class FPN50(nn.Module):

    def __init__(self, pretrained_path):
        super(FPN50, self).__init__()
        self.backbone = ResNet50Stages(pretrained_path)

        self.lateral_layer1 = nn.Conv2d(2048, 256, 1)
        self.lateral_layer2 = nn.Conv2d(1024, 256, 1)
        self.lateral_layer3 = nn.Conv2d(512,  256, 1)

        self.conv6 = nn.Conv2d(2048, 256, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(256,  256, 3, padding=1, stride=2)
        self.relu = nn.ReLU(inplace=True)

        self._weight_initialize()

    def _weight_initialize(self):

        self.lateral_layer1.weight.data.normal_(std=0.01)
        self.lateral_layer1.bias.data.fill_(0.0)

        self.lateral_layer2.weight.data.normal_(std=0.01)
        self.lateral_layer2.bias.data.fill_(0.0)

        self.lateral_layer3.weight.data.normal_(std=0.01)
        self.lateral_layer3.bias.data.fill_(0.0)

        self.conv6.weight.data.normal_(std=0.01)
        self.conv6.bias.data.fill_(0.0)

        self.conv7.weight.data.normal_(std=0.01)
        self.conv7.bias.data.fill_(0.0)

    def upsample_add(self, x, y):
        _, _, h, w = y.size()
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        # c2: 64*4=256 c3: 128*4=512 c4: 256*4=1024 c5: 512*4=2048

        p5 = self.lateral_layer1(c5)

        p4 = self.lateral_layer2(c4)
        p4 = self.upsample_add(p5, p4)

        p3 = self.lateral_layer3(c3)
        p3 = self.upsample_add(p4, p3)

        p6 = self.conv6(c5)
        p7 = self.conv7(self.relu(p6))

        return p3, p4, p5, p6, p7
