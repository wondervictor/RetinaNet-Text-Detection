"""
Basemodel: ResNet

"""

import torch
from torchvision.models import resnet
import torch.nn as nn

Bottleneck = resnet.Bottleneck


class ResNet50Stages(nn.Module):

    def __init__(self, pretrained_path):
        super(ResNet50Stages, self).__init__()
        self.inplanes = 64
        self.stages = [3, 4, 6, 3]
        self.mid_outputs = [64, 128, 256, 512]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, self.mid_outputs[0], self.stages[0])
        self.layer2 = self._make_layer(Bottleneck, self.mid_outputs[1], self.stages[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, self.mid_outputs[2], self.stages[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, self.mid_outputs[3], self.stages[3], stride=2)

        # self.load_state_dict(torch.load(pretrained_path))
        self.load_pretrained(pretrained_path)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        pass

    def load_pretrained(self, mpath):
        
        pretrained_dict = torch.load(mpath)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        self.load_state_dict(pretrained_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x2, x3, x4]


class ResNet50(nn.Module):

    def __init__(self, pretrained_path):
        super(ResNet50, self).__init__()
        self.layers = ResNet50Stages(pretrained_path)

    def forward(self, x):
        return self.layers(x)[-1]


class ResNet50C4(nn.Module):

    def __init__(self, pretrained_path):
        super(ResNet50C4, self).__init__()
        self.inplanes = 64
        self.stages = [3, 4, 6]
        self.mid_outputs = [64, 128, 256, 512]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, self.mid_outputs[0], self.stages[0])
        self.layer2 = self._make_layer(Bottleneck, self.mid_outputs[1], self.stages[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, self.mid_outputs[2], self.stages[2], stride=2)

        # self.load_state_dict(torch.load(pretrained_path))
        self.load_pretrained(pretrained_path)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        pass

    def load_pretrained(self, mpath):

        pretrained_dict = torch.load(mpath)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        self.load_state_dict(pretrained_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return x3

