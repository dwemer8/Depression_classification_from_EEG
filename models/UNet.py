# https://github.com/milesial/Pytorch-UNet
# """ Full assembly of the parts to form the complete network """
import torch.nn as nn

from .modules import Down, DoubleConv, Up_with_sc, OutDoubleConv

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, c1=64, c2=128, c3=256, c_neck=512):
        super(UNet1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.down1 = Down(n_channels, c1, kernel_size=7) #64*256
        self.down2 = Down(c1, c2, kernel_size=7) #128*128
        self.down3 = Down(c2, c3,kernel_size=5) #256*64
        self.neck = DoubleConv(c3, c_neck, kernel_size=3) #512*64
        self.up1 = Up_with_sc(c_neck, c3, kernel_size=3, concat=False) #256*128
        self.up2 = Up_with_sc(c3+c3, c2, kernel_size=3) #128*256
        self.up3 = Up_with_sc(c2+c2, c1, kernel_size=3) #64*512
        self.outc = OutDoubleConv(c1+n_channels, n_classes, kernel_size=1) #n_classes*512

    def forward(self, x):
        x, x1_skip = self.down1(x) 
        x, x2_skip = self.down2(x) 
        x, x3_skip = self.down3(x) 
        x = self.neck(x)
        x = self.up1(x)
        x = self.up2(x, x3_skip)
        x = self.up3(x, x2_skip)
        logits = self.outc(torch.cat([x1_skip, x], axis=1)) #B * C * H * W
        return logits
    
    def reconstruct(self, x):
        return self.forward(x)
    
    def encode(self, x):
        x, x1_skip = self.down1(x) 
        x, x2_skip = self.down2(x) 
        x, x3_skip = self.down3(x) 
        x = self.neck(x)
        return x
    
    def decode(self, x):
        raise NotImplemented