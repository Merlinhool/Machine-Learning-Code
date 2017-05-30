import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.nn import init


__all__ = ['ResNet', 'resnet47', 'resnet_bottleneck']

class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        #TODO do different thing when dim increasing is need
        planes = out_planes / 4

        #1x1 conv
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

        #3x3 conv
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        #1x1 conv
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, bias=False)

        self.downsample = downsample
        
    def forward(self, x):
        if self.downsample is None:
            residual = x
            out = self.bn1(x)
            out = self.relu1(out)
        else:
            x = self.bn1(x)
            x = self.relu1(x)
            out = x
            residual = self.downsample(x)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        out += residual
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, planes, num_classes=10):
        #TODO super.__init__()
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, planes[0], kernel_size=3, stride=1, padding=1,
                               bias=False)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], planes[0], planes[1], stride=1)
        self.layer2 = self._make_layer(block, layers[1], planes[1], planes[2], stride=2)
        self.layer3 = self._make_layer(block, layers[2], planes[2], planes[3], stride=2)

        self.bn = nn.BatchNorm2d(planes[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(planes[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, block, blocks, in_planes, out_planes, stride=1):
        downsample = None
        if stride != 1 or in_planes != out_planes:
            downsample = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

        layers = []
        layers.append(block(in_planes, out_planes, stride, downsample))
        for i in range(1, blocks):
            layers.append(block(out_planes, out_planes, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet47(**kwargs):
    print '47-layer ResNet (Bottleneck)'
    model = ResNet(Bottleneck, [5, 5, 5], [16, 64, 128, 256], **kwargs)
    print 'modules'
    for i in model.modules():
        print i
    return model

def resnet164(**kwargs):
    print '164-layer ResNet (Bottleneck)'
    model = ResNet(Bottleneck, [18, 18, 18], [16, 64, 128, 256], **kwargs)
    print 'modules'
    for i in model.modules():
        print i
    return model

def resnet_bottleneck(depth, **kwargs):
    print str(depth) + '-layer ResNet\nBottleneck && 1x1 conv for downsample'
    assert (depth - 2) % 9 == 0
    layers = [(depth-2) / 9] * 3
    model = ResNet(Bottleneck, layers, [16, 64, 128, 256], **kwargs)
    print 'modules'
    for i in model.modules():
        print i
    return model
