#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:30:53 2019

@author: liqinying
"""
import torch.nn as nn
from .utils import nn_utils

class Pose_Enc(nn.Module):
    
    """
    Pose Encoder, of which the architecture is Unet-like.
    
    Arguments
        in_channels: The channels of each input. If input is RGB colored image, then in_channels=3
        n_features: The number of feature maps as output; The parameter K in the paper
        norm: The mode of normalization layer. Should be either "Batch" or "Instance"
    """
    
    def __init__(self, in_channels, n_features, norm=None, bilinear=False):
        super(Pose_Enc, self).__init__()
        self.norm=norm

        self.inc = nn_utils.inconv(in_channels, 64, norm=norm)
        self.down1 = nn_utils.down(64, 128, norm=self.norm)
        self.down2 = nn_utils.down(128, 256, norm=self.norm)
        self.down3 = nn_utils.down(256, 512, norm=self.norm)
        self.down4 = nn_utils.down(512, 512,norm=self.norm)
        self.up1 = nn_utils.up(1024, 256, bilinear=bilinear, norm=None)
        self.up2 = nn_utils.up(512, 128, bilinear=bilinear, norm=None)
        self.up3 = nn_utils.up(256, 64,bilinear=bilinear, norm=None)
        self.up4 = nn_utils.up(128, 64,bilinear=bilinear, norm=None)
        self.outc = nn_utils.outconv(64, n_features)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
    
class App_Enc(nn.Module):
  """
    Appearance Encoder, of which the architecture is Unet-like.
    
    Arguments
        n_features: The channels of each input. If input is RGB colored image, then in_channels=3
        n_appearance: The number of feature maps as output; The parameter K in the paper
        norm: The mode of normalization layer. Should be either "Batch" or "Instance"
  """  
  
  def __init__(self, n_features, n_appearance, norm=None):
    super(App_Enc, self).__init__()

    self.inc=nn_utils.inconv(n_features,8, norm=norm)
    self.down1=nn_utils.down(8, 16, norm=norm)
    self.mid=nn_utils.down(16, 16, norm=norm)
    self.up1=nn_utils.up(32, 8, norm=norm)
    self.outc=nn_utils.outconv(8, n_appearance)
  
  def forward(self,x):
    x1=self.inc(x)
    x2=self.down1(x1)
    x3=self.mid(x2)
    x=self.up1(x3,x2)
    x=self.outc(x)

    return x

class MaskNet(nn.Module):
    """
    Mask Networks, of which the architecure is Unet-like
    Arguments
        n_features: of the semantic
        n_filter: The number of filters used at each layer
        kernel_size: the size of kernel (filter).
    """
    
    def __init__(self, n_features, norm=None):
        super(MaskNet, self).__init__()
        self.inc=nn_utils.inconv(n_features, 32, norm=norm)
        self.down1=nn_utils.down(32,32, norm=norm)
        self.down2=nn_utils.down(32,32, norm=norm)
        self.down3=nn_utils.down(32,32, norm=norm)
        #have to think about how to concatanete this...
        self.up1=nn_utils.up(32, 32, norm=norm)
        self.up2=nn_utils.up(32, 32, norm=norm)
        pass
    
    def forward(self, x):
        pass
    