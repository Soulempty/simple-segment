import torch.nn as nn
from torchvision import models
from models.registry import BACKBONES

def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)

class ConvX(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, dilation=(1, 1), group=1, bias=False, with_bn=True, with_act=True):
        super(ConvX, self).__init__()
        if padding==None:
            padding = kernel_size//2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=group, bias=bias)
        self.with_bn = with_bn
        if self.with_bn:
            self.bn = nn.BatchNorm2d(out_channels)     
        self.with_act = with_act
        if self.with_act:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.with_bn:
            x = self.bn(x)
        if self.with_act:
            x = self.relu(x)
        return x  
            
class SegHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegHead, self).__init__()
        self.conv = ConvX(in_channels, out_channels, 1, bias=True, with_bn=False,with_act=False)
    def forward(self, x):
        x = self.conv(x)
        return x
    
class Segment_(nn.Module):
    def __init__(self,model):
        super(Segment_, self).__init__()
        self.model = model

    def forward(self, inputs):
        x = self.model(inputs)[0]
        x = x.argmax(1, keepdims=True)
        return x
    
@BACKBONES.register_module  
class ResNet18(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        model = models.resnet18() 
        self.filters = [64,128,256,512]
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x))) # 1/2
        layer0 = self.maxpool(x)      
        layer1 = self.layer1(layer0)  # 1/4
        layer2 = self.layer2(layer1)  # 1/8
        layer3 = self.layer3(layer2)  # 1/16
        layer4 = self.layer4(layer3)  # 1/32
        return layer1,layer2,layer3,layer4
    
@BACKBONES.register_module 
class ResNet34(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        model = models.resnet18() 
        self.filters = [64,128,256,512]
        if pretrained:
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x))) # 1/2
        layer0 = self.maxpool(x)      
        layer1 = self.layer1(layer0)  # 1/4
        layer2 = self.layer2(layer1)  # 1/8
        layer3 = self.layer3(layer2)  # 1/16
        layer4 = self.layer4(layer3)  # 1/32
        return layer1,layer2,layer3,layer4

@BACKBONES.register_module    
class ResNet50(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        model = models.resnet50() 
        self.filters = [256,512,1024,2048]
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x))) # 1/2
        layer0 = self.maxpool(x)      
        layer1 = self.layer1(layer0)  # 1/4
        layer2 = self.layer2(layer1)  # 1/8
        layer3 = self.layer3(layer2)  # 1/16
        layer4 = self.layer4(layer3)  # 1/32
        return layer1,layer2,layer3,layer4