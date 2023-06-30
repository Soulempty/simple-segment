import torch
import torch.nn as nn
import torch.nn.functional as F
from models.registry import MODELS

from .module import init_weight, ConvX, ResNet18, ResNet34, ResNet50, SegHead

class DAPPM(nn.Module): 
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=2, padding=1), # 1/2
                                    ConvX(inplanes, branch_planes, 1))
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=4, padding=1), # 1/4
                                    ConvX(inplanes, branch_planes, 1))
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=8, padding=1), # 1/8
                                    ConvX(inplanes, branch_planes, 1))
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    ConvX(inplanes, branch_planes, 1))
        self.scale0 = ConvX(inplanes, branch_planes, 1)  
        self.process1 = ConvX(branch_planes, branch_planes, 3)  
        self.process2 = ConvX(branch_planes, branch_planes, 3)  
        self.process3 = ConvX(branch_planes, branch_planes, 3)  
        self.process4 = ConvX(branch_planes, branch_planes, 3) 
        self.shortcut = ConvX(inplanes, outplanes, 1)
        self.compression = ConvX(branch_planes * 5, outplanes, 1) 

    def forward(self, x):

        width = x.shape[-1]
        height = x.shape[-2]        
        scale0 = self.scale0(x)
        b1 = self.process1((F.interpolate(self.scale1(x),size=[height, width],mode='bilinear')+scale0))
        b2 = self.process2((F.interpolate(self.scale2(x),size=[height, width],mode='bilinear')+b1))
        b3 = self.process3((F.interpolate(self.scale3(x),size=[height, width],mode='bilinear')+b2))
        b4 = self.process3((F.interpolate(self.scale4(x),size=[height, width],mode='bilinear')+b3))
        
        out = self.compression(torch.cat([scale0,b1,b2,b3,b4], 1)) + self.shortcut(x)
        return out 
    

class Decoder(nn.Module):
    def __init__(self, filters=[64, 128, 256, 512]):
        super(Decoder, self).__init__()
        self.cat_channel = filters[0]
        self.up_channel = self.cat_channel*2

        self.upsample2 = nn.Upsample(scale_factor=2,mode='bilinear')
        self.dappm = DAPPM(filters[3],self.up_channel,self.up_channel)

        # stage4-d
        self.h4_d4 = ConvX(filters[2],self.cat_channel,3)
        self.spp_d4 = ConvX(self.up_channel,self.cat_channel,3)
        self.conv_d4 = ConvX(self.up_channel,self.up_channel,3)

        # stage3-d
        self.h3_d3 = ConvX(filters[1],self.cat_channel,3)
        self.d4_d3 = ConvX(self.up_channel,self.cat_channel,3)
        self.conv_d3 = ConvX(self.up_channel,self.up_channel,3)

        # stage2-d
        self.h2_d2 = ConvX(filters[0],self.cat_channel,3)
        self.d3_d2 = ConvX(self.up_channel,self.cat_channel,3)
        self.conv_d2 = ConvX(self.up_channel,self.up_channel,3)
        init_weight(self)
        
    def forward(self,feats):
        h2,h3,h4,d5 = feats
        dappm = self.dappm(d5)

        #d4
        h4_d4 = self.h4_d4(h4)
        spp_d4 = self.spp_d4(self.upsample2(dappm))
        d4 = self.conv_d4(torch.cat([h4_d4,spp_d4],1))

        #d3
        h3_d3 = self.h3_d3(h3)
        d4_d3 = self.d4_d3(self.upsample2(d4))
        d3 = self.conv_d3(torch.cat([h3_d3,d4_d3],1))

        #d2
        h2_d2 = self.h2_d2(h2)
        d3_d2 = self.d3_d2(self.upsample2(d3))
        d2 = self.conv_d2(torch.cat([h2_d2,d3_d2],1))
        
        return [d2,d3,d4,dappm]

@MODELS.register_module
class UNet(nn.Module):
    def __init__(self, num_classes=19, backbone=None, deep_sup=False):
        super(UNet, self).__init__()
        self.deepsup = deep_sup
        if backbone == None:
            backbone = ResNet18()
        self.encoder = backbone # 1/4,1/8,1/16,1/32
        filters = self.encoder.filters
        self.decoder = Decoder(filters)

        up_channel = filters[0]*2
        self.cls_d2 = SegHead(up_channel,num_classes)
        self.upsample = nn.Upsample(scale_factor=4,mode='bilinear')
        if deep_sup:
            self.cls_d3 = SegHead(up_channel,num_classes)
            self.cls_d4 = SegHead(up_channel,num_classes)
            self.cls_d5 = SegHead(up_channel,num_classes)
            self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
            self.upsample5 = nn.Upsample(scale_factor=8, mode='bilinear')

    def forward(self,x):
        feats = self.encoder(x)
        d2,d3,d4,d5 = self.decoder(feats)
        cls_d2 = self.upsample(self.cls_d2(d2))
        if self.deepsup:
            cls_d3 = self.upsample(self.cls_d3(self.upsample3(d3)))
            cls_d4 = self.upsample(self.cls_d4(self.upsample4(d4)))
            cls_d5 = self.upsample(self.cls_d5(self.upsample5(d5)))
            return cls_d2,cls_d3,cls_d4,cls_d5
        return [cls_d2]
    
if __name__ == "__main__":
    img = torch.randn([1,3,832,832]).cuda()
    model = UNet(20).cuda()
    model.eval()
    from tqdm import tqdm
    for i in tqdm(range(1000)):
        out = model(img)
        print(out[0].size())

