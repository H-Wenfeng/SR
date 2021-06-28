import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import cv2
import numpy as np
sys.path.append("..") 
import matplotlib.pyplot as plt
from src.net.dcn.deform_conv import ModulatedDeformConvPack as DCN


class Efblock(nn.Module):
    def __init__(self, Cin, Cout, kernel, stride, exp):
        super(Efblock, self).__init__()
        padding = (kernel - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(Cin, exp, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
        )     
        self.conv2 = nn.Conv2d(exp, exp, kernel, stride, padding, groups= exp // 8, bias=False)
        self.conv3 = nn.Sequential(    
            nn.ReLU(inplace=True),
            nn.Conv2d(exp, Cout, 1, 1, 0, bias=False,groups=8),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(Cin, Cout, 1, 1, 0),
            # nn.BatchNorm2d(Cout), 
            nn.ReLU(inplace=True)
        )
    def forward(self, x):

        out1 = self.conv1(x)
        out1 = self.conv2(out1) + out1
        out1 = self.conv3(out1)

        return out1 + self.conv1x1(x)

class RF(nn.Module):
    def __init__(self, Cin, scale):
        super(RF, self).__init__()
        self.scale = scale
        if self.scale == 5:
            self.d = 2
        else:
            self.d = 3


        self.dcnblock1 = nn.Sequential(
        nn.Conv2d(Cin, Cin, 3, 1, self.d, dilation=self.d),
        nn.LeakyReLU(0.05)
        )
        self.dcnblock2 = nn.Sequential(
        DCN(Cin, Cin, 3, 1,1),
        nn.LeakyReLU(0.05)
        )


    def forward(self, x):
            out = self.dcnblock1(x) + x
            res = self.dcnblock2(out) + out   
            return res + x


class Efblocks(nn.Module):
    def __init__(self,):
        super (Efblocks,self).__init__()
        self.block1 = Efblock(3, 64, 3, 1, 32)
        self.block2 = Efblock(64, 64, 3, 1, 64)

    def forward(self,x):
        out = self.block1(x)
        out = self.block2(out)
        
        return out



class HMblock(nn.Module):
    def __init__(self):
        super(HMblock, self).__init__()
        self.RF3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05),
            DCN( 128, 128, 3, 1, 1 ),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05)
        )
        self.RF5 = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=1, padding=0),
            RF(48, 5),
            nn.Conv2d(48, 64, kernel_size=1, padding=0),
            nn.LeakyReLU(0.05)
        )
        self.RF7 = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=1, padding=0),
            RF(48, 8),
            nn.Conv2d(48, 64, kernel_size=1, padding=0),
            nn.LeakyReLU(0.05)
        )
 
        self.DS = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(128, 80, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(80, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05)
        )
        self.compression = nn.Sequential(
        nn.Conv2d(128, 64, kernel_size=1),

        )
        
    def forward(self, x):
        residual = x
 
        resca = residual
        x1 = self.RF3(x) + resca
        x2 = self.RF5(x) + resca
        x3 = self.RF7(x) + resca

        x = x1 + x2 + x3


        slice_1 = x
        slice_2 = x
        x = self.DS(slice_1)
        x = x + torch.cat((resca, slice_2), 1)
        x = self.compression(x)
        return x 

class HMSF(nn.Module):
    def __init__(self):
        super(HMSF, self).__init__()
        self.Ef = Efblocks()
        self.HMblock = HMblock()

        self.DRblock = nn.Sequential(
            nn.Conv2d(64, 48, 1, 1),          
            nn.LeakyReLU(0.05)

            )
        self.lrelu = nn.LeakyReLU(0.05)
        self.pixel_shuffle = nn.PixelShuffle(4)


    def forward(self, x):    
        fea = self.Ef(x)
      

        fea_ref = fea
        fea_tot = self.HMblock(fea_ref)
        fea_tot = self.HMblock(fea_tot)
        fea_tot = self.HMblock(fea_tot)
        fea_tot = self.HMblock(fea_tot)
        out = fea_tot 
        out = self.DRblock(out)
        x_bicubic = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        out =  self.pixel_shuffle(out)
        out = out + x_bicubic
        return out 


