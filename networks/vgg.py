#########   TO DO###########
#
#MSE loss
#to be put in the train.py...

from collections import namedtuple

import torch.nn as nn
from torchvision import models


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.block1 = nn.Sequential()
        self.block2 = nn.Sequential()
        self.block3 = nn.Sequential()
        self.block4 = nn.Sequential()
        
        for x in range(4): #relu1_2
            self.block1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9): #relu2_2
            self.block2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 14): #relu3_2
            self.block3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 21): #relu4_2
            self.block4.add_module(str(x), vgg_pretrained_features[x])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

def forward(self, X):
    h = self.block1(X)
    h_relu1_2 = h
        h = self.block2(h)
        h_relu2_2 = h
        h = self.block3(h)
        h_relu3_3 = h
        h = self.block4(h)
        h_relu4_3 = h
        vgg19_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_2', 'relu4_2'])
        out = vgg19_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
