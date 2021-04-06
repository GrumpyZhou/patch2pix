"""
This script is an adapted version of 
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py 
The goal is to keep ResNet* as only feature extractor, 
so the code can be used independent of the types of specific tasks,
i.e., classification or regression. 
"""
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    PRETRAINED_URLs = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }
    
    def __init__(self):
        super().__init__()
        
    def _build_model(self, block, layers):
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

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

    def forward(self, x, early_feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if early_feat:
            return x
        x = self.layer4(x)
        return x
    
    def forward_all(self, x, feat_list=[], early_feat=True):
        feat_list.append(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        feat_list.append(x)
        
        x = self.maxpool(x)
        x = self.layer1(x)
        feat_list.append(x)
        
        x = self.layer2(x)
        feat_list.append(x)
        
        x = self.layer3(x)
        feat_list.append(x)
        
        if not early_feat:
            x = self.layer4(x)
            feat_list.append(x)
    
    def load_pretrained_(self, ignore='fc'):
        print('Initialize ResNet using pretrained model from {}'.format(self.pretrained_url))
        state_dict = model_zoo.load_url(self.pretrained_url)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if ignore in k:
                continue
            new_state_dict[k] = v
        self.load_state_dict(new_state_dict)

    def change_stride(self, target='layer3'):
        layer = getattr(self, target)
        layer[0].conv1.stride = (1, 1)
        layer[0].conv2.stride = (1, 1)
        layer[0].downsample[0].stride = (1, 1) 

class ResNet34(ResNet):
    def __init__(self):
        super().__init__()
        self.pretrained_url = self.PRETRAINED_URLs['resnet34']
        self._build_model(BasicBlock, [3, 4, 6, 3])

class ResNet50(ResNet):
    def __init__(self):
        super().__init__()
        self.pretrained_url = self.PRETRAINED_URLs['resnet50']
        self._build_model(Bottleneck, [3, 4, 6, 3])
        
class ResNet101(ResNet):
    def __init__(self):
        super().__init__()
        self.pretrained_url = self.PRETRAINED_URLs['resnet101']
        self._build_model(Bottleneck, [3, 4, 23, 3])
