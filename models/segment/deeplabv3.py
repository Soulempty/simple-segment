from collections import OrderedDict
from typing import Dict, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from models.registry import MODELS
from .module import init_weight, ConvX, ResNet18, ResNet34, SegHead


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            ConvX(in_channels, out_channels, 1))

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP,self).__init__()
        modules = []
        modules.append(ConvX(in_channels, out_channels, 1))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ConvX(in_channels, out_channels,3,padding=rate, dilation=rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = ConvX(len(self.convs) * out_channels, out_channels, 1)

    def forward(self, x):
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)
    
class DeepLabHead(nn.Module):
    def __init__(self, in_channels, inter_channel=64):
        super(DeepLabHead,self).__init__()
        self.aspp = ASPP(in_channels, [3,5,7], inter_channel) 
        self.lconv = ConvX(inter_channel, inter_channel, 1)
        self.mconv = ConvX(inter_channel*2,inter_channel,3) 
        init_weight(self)

    def forward(self, features):
        feat4,feat8,feat16,feat32 = features
        size = feat4.shape[-2:]
        hfeat = self.aspp(feat32)
        hfeat = F.interpolate(hfeat, size=size, mode="bilinear", align_corners=False)
        lfeat = self.lconv(feat4)
        feat = self.mconv(torch.cat([hfeat,lfeat],dim=1))
        return feat


@MODELS.register_module
class DeepLab(nn.Module):

    def __init__(self, num_classes,backbone=None):
        super(DeepLab, self).__init__()
        if backbone == None:
            backbone = ResNet18()
        self.backbone = backbone
        channels = self.backbone.filters    
        self.head = DeepLabHead(channels[-1],channels[0])
        self.seghead = SegHead(channels[0],num_classes)
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        feat = self.head(features)
        logits = self.seghead(feat)
        logits = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)
        return [logits]
 
if __name__ == "__main__":
    img = torch.randn([1,3,832,832]).cuda()
    model = DeepLab(20).cuda()
    model.eval()
    from tqdm import tqdm
    for i in tqdm(range(1000)):
        out = model(img)
        print(out[0].size())