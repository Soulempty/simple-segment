import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from models.registry import MODELS
from .module import ConvX,ResNet18,SegHead

@MODELS.register_module   
class PAN(nn.Module):
    def __init__(self, num_classes):
        super(PAN,self).__init__()
        self.encoder = ResNet18() # [64,128,256,512]
        channels = self.encoder.filters
        out_channel = channels[3]//8
        
        self.fpa = FPAModule(channels[3],out_channel)
        self.gau3 = GAUModule(channels[2], out_channel)
        self.gau2 = GAUModule(channels[1], out_channel)
        self.gau1 = GAUModule(channels[0], out_channel)
        self.seghead = SegHead(out_channel,num_classes)
                
    def forward(self,x):
        x1,x2,x3,x4 = self.encoder(x) # 1/4 1/8 1/16 1/32
        x4 = self.fpa(x4)        # 1/32
        x3 = self.gau3(x3,x4)    # 1/16
        x2 = self.gau2(x2,x3)    # 1/8
        x1 = self.gau1(x1,x2)    # 1/4
        logits = F.interpolate(self.seghead(x1),size=x.size()[2:],mode='bilinear')
        return [logits]   
		
class FPAModule(nn.Module):
	
    def __init__(self, in_channel, out_channel):
        super(FPAModule, self).__init__()
        
        self.gl_branch = nn.Sequential(nn.AdaptiveAvgPool2d(1),ConvX(in_channel, out_channel, 1))
        self.conv1 = ConvX(in_channel, out_channel, 1)		
        self.down1 = ConvX(in_channel, out_channel, 7, 2)
        self.down2 = ConvX(out_channel, out_channel, 5, 2)
        self.down3 = nn.Sequential(ConvX(out_channel, out_channel, 3, 2),ConvX(out_channel, out_channel, 3, 1))
        
        self.conv2 = ConvX(out_channel, out_channel, 5)
        self.conv3 = ConvX(out_channel, out_channel, 7) 
	
    def forward(self, x):
        gl = self.gl_branch(x)
        x_ = self.conv1(x)
        
        x1 = self.down1(x) #  2
        x2 = self.down2(x1) # 4
        x3 = self.down3(x2) # 8
    
        up3 = F.interpolate(x3,size=x2.size()[2:],mode='bilinear')
        
        x2 = self.conv2(x2)
        up2 = F.interpolate(x2+up3,size=x1.size()[2:],mode='bilinear')
        
        x1 = self.conv3(x1)
        up1 = F.interpolate(x1+up2,size=x_.size()[2:],mode='bilinear') 
        
        x = x_*up1+gl
        return x


class GAUModule(nn.Module):
    def __init__(self,low_channel,out_channel):   #
        super(GAUModule, self).__init__()
        
        self.conv1 = ConvX(out_channel, out_channel, 1)
        self.conv2 = ConvX(low_channel,out_channel,3)
        self.avg = nn.AdaptiveAvgPool2d(1)
		
    def forward(self,x,y):
        x = self.conv2(x)
        y = self.conv1(y)
        y1 = self.avg(y)
        y = F.interpolate(y,size=x.size()[2:],mode='bilinear')
        z = x*y1 + y	
        return z

if __name__ == '__main__':   
    dummy_in = torch.randn(1, 3,384,384).cuda()
    model = PAN(20).cuda()
    model.eval()
    with torch.no_grad():
        output = model(dummy_in)   
        print("output size:",output[0].size()) 
    