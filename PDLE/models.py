import torch
import torch.nn as nn
import numpy as np 
import math
import torch.nn.functional as F 


class CNN_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels):
        super(CNN_Encoder, self).__init__()
        self.layer1 = nn.Sequential(
                            nn.Conv2d(in_channels, hid_channels, 3, padding=1),
                            nn.BatchNorm2d(hid_channels),
                            nn.ReLU(),
                            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                            nn.Conv2d(hid_channels, hid_channels, 3, padding=1),
                            nn.BatchNorm2d(hid_channels),
                            nn.ReLU(),
                            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                            nn.Conv2d(hid_channels, hid_channels, 3, padding=1),
                            nn.BatchNorm2d(hid_channels),
                            nn.ReLU(),
                            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
                            nn.Conv2d(hid_channels, out_channels, 3, padding=1),
                            nn.BatchNorm2d(hid_channels),
                            nn.ReLU(),
                            nn.MaxPool2d(2))
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, inputs):
        output = self.layer1(inputs)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return output.view(output.size(0),-1)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP,self).__init__()
        self.fc_out = nn.Linear(in_dim, out_dim)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal(m.weight)

    def forward(self, inputs):
        h = inputs.view(inputs.size(0),-1)
        output = self.fc_out(h)
        return output

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    
    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(lambda x:F.pad(x[:,:,::2, ::2], (0,0,0,0,planes//4, planes//4),'constant',0))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet32_Encoder(nn.Module):
    def __init__(self, block, num_blocks = [5,5,5], num_classes = 10):
        super(ResNet32_Encoder, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # self.linear = nn.Linear(64, num_classes)
        self._init_weight()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return out


class Resnet32(nn.Module):
    def __init__(self):
        super(Resnet32, self).__init__()
        self.encoder = resnet32()
        self.fc = MLP(in_dim=64, out_dim=10)

    def forward(self, x):
        features = self.encoder(x)
        return self.fc(features)

def resnet32():
    return ResNet32_Encoder(BasicBlock)


