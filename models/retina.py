"""
RetinaNet Model
backbone: resnet50 + FPN
"""

import torch
import torch.nn as nn
import numpy as np
from models import fpn


class RetinaNetHead(nn.Module):

    def __init__(self, num_classes, num_anchors):
        super(RetinaNetHead, self).__init__()
        self.num_classes = num_classes

        self.cls_branch = nn.Sequential(
            nn.Conv2d(256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.cls_score = nn.Conv2d(256, out_channels=num_classes*num_anchors, kernel_size=3, stride=1, padding=1)

        self.bbox_branch = nn.Sequential(
            nn.Conv2d(256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels=num_anchors*4, kernel_size=3, stride=1, padding=1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.cls_branch.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0)

        for m in self.bbox_branch.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0)

        self.cls_score.weight.data.normal_(0, 0.01)
        pi = 0.01
        self.cls_score.bias.data.fill_(-np.log((1 - pi) / pi))

    def forward(self, x):
        bbox_output = self.bbox_branch(x)
        bbox_output = bbox_output.permute(0, 2, 3, 1).contiguous().view(x.size()[0], -1, 4)
        cls_output = self.cls_score(self.cls_branch(x))
        cls_output = cls_output.permute(0, 2, 3, 1).contiguous().view(x.size()[0], -1, self.num_classes)
        return cls_output, bbox_output


class RetinaNet(nn.Module):

    def __init__(self, num_classes, num_anchors, pretrained_path):
        super(RetinaNet, self).__init__()
        self.fpn = fpn.FPN50(pretrained_path)
        self.head = RetinaNetHead(num_classes, num_anchors)

    def forward(self, x):
        # [P3, P4, P5, P6, P7]
        # stride: [8, 16, 32, 64, 128]
        feature_pyramids = self.fpn(x)
        cls_outputs = []
        bbox_outputs = []
        for fp in feature_pyramids:
            cls_output, bbox_output = self.head(fp)
            cls_outputs.append(cls_output)
            bbox_outputs.append(bbox_output)

        cls_outputs = torch.cat(cls_outputs, dim=1)
        bbox_outputs = torch.cat(bbox_outputs, dim=1)

        return cls_outputs, bbox_outputs



