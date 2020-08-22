import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torchsummary import summary
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from medpy.io import load
from skimage.transform import resize
from multiprocessing.dummy import Pool as ThreadPool
# pytorch implementation of HDenseUNet
# the base file is from the repo https://github.com/thangylvp/HDenseUet/blob/master/HDenseUnet.py
# the repo contains encoer decoder format instead of UNet

device = 'cuda'

class Scale(nn.Module): 
    def __init__(self, num_feature): 
        super(Scale, self).__init__() 
        self.num_feature = num_feature 
        self.gamma = nn.Parameter(torch.ones(num_feature), requires_grad=True) 
        self.beta = nn.Parameter(torch.zeros(num_feature), requires_grad=True) 
    def forward(self, x): 
        y = torch.zeros(x.shape, dtype= x.dtype, device= x.device)
        for i in range(self.num_feature): 
            y[:, i, :, :] = x[:, i, :, :].clone() * self.gamma[i] + self.beta[i] 
        return y

class dense_block(nn.Module):
    def __init__(self, nb_layers, nb_filter, growth_rate, dropout_rate=0, weight_decay=1e-4, grow_nb_filters=True):
        super(dense_block, self).__init__()
        for i in range(nb_layers):
            layer = conv_block(nb_filter + i * growth_rate, growth_rate, dropout_rate)
            self.add_module('denseLayer%d' % (i + 1), layer)

    def forward(self, x):
        features = [x]
        for name, layer in self.named_children():
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)

# some problem with the transition class, def forward absent?
class _Transition(nn.Sequential):
    def __init__(self, num_input, num_output, drop=0):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input))
        self.add_module('scale', Scale(num_input))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv2d', nn.Conv2d(num_input, num_output, (1, 1), bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size= 2, stride= 2))
    # added this part as it was not present in the original code
    def forward(self, x):
        out = self.conv2d(self.relu(self.scale(self.norm(x))))
        if (self.drop > 0):
            out = F.dropout(out, p= self.drop)
        return F.avg_pool2d(out, 2)



class conv_block(nn.Sequential):
    def __init__(self, nb_inp_fea, growth_rate, dropout_rate=0, weight_decay=1e-4):
        super(conv_block, self).__init__()
        eps = 1.1e-5
        self.drop = dropout_rate
        self.add_module('norm1', nn.BatchNorm2d(nb_inp_fea, eps= eps, momentum= 1))
        self.add_module('scale1', Scale(nb_inp_fea))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv2d1', nn.Conv2d(nb_inp_fea, 4 * growth_rate, (1, 1), bias=False))
        self.add_module('norm2', nn.BatchNorm2d(4 * growth_rate, eps= eps, momentum= 1))
        self.add_module('scale2', Scale(4 * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace= True))
        self.add_module('conv2d2', nn.Conv2d(4 * growth_rate, growth_rate, (3, 3), padding = (1,1), bias= False))
        
    
    def forward(self, x):
        out = self.conv2d1(self.relu1(self.scale1(self.norm1(x))))
        if (self.drop > 0):
            out = F.dropout(out, p= self.drop)
        
        out = self.conv2d2(self.relu2(self.scale2(self.norm2(out))))
        if (self.drop > 0):
            out = F.dropout(out, p= self.drop)

        return out


class denseUnet(nn.Module):
    def __init__(self, growth_rate=48, block_config=(6, 12, 36, 24), num_init_features=96, drop_rate=0, weight_decay=1e-4, num_classes=1000):
        super(denseUnet, self).__init__()
        nb_filter = num_init_features
        eps = 1.1e-5
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, nb_filter, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(nb_filter, eps= eps)),
            ('scale0', Scale(nb_filter)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        for i, num_layer in enumerate(block_config):
            block = dense_block(num_layer, nb_filter, growth_rate, drop_rate)
            nb_filter += num_layer * growth_rate
            self.features.add_module('denseblock%d' % (i + 1), block)
            if i != len(block_config) - 1:
                trans = _Transition(nb_filter, nb_filter // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                nb_filter = nb_filter // 2
                
        self.features.add_module('norm5', nn.BatchNorm2d(nb_filter, eps= eps, momentum= 1))
        self.features.add_module('scale5', Scale(nb_filter))
        self.features.add_module('relu5', nn.ReLU(inplace= True))

        self.decode = nn.Sequential(OrderedDict([
            ('up0', nn.Upsample(scale_factor=2)),
            ('conv2d0', nn.Conv2d(nb_filter, 768, (3, 3), padding= 1)),
            ('bn0', nn.BatchNorm2d(768, momentum= 1)), 
            ('ac0', nn.ReLU(inplace=True)),
            
            ('up1', nn.Upsample(scale_factor=2)),
            ('conv2d1', nn.Conv2d(768, 384, (3, 3), padding= 1)),
            ('bn1', nn.BatchNorm2d(384, momentum= 1)), 
            ('ac1', nn.ReLU(inplace=True)),

            ('up2', nn.Upsample(scale_factor=2)),
            ('conv2d2', nn.Conv2d(384, 96, (3, 3), padding= 1)),
            ('bn2', nn.BatchNorm2d(96, momentum= 1)), 
            ('ac2', nn.ReLU(inplace=True)),

            ('up3', nn.Upsample(scale_factor=2)),
            ('conv2d3', nn.Conv2d(96, 96, (3, 3), padding= 1)),
            ('bn3', nn.BatchNorm2d(96, momentum= 1)), 
            ('ac3', nn.ReLU(inplace=True)),

            ('up4', nn.Upsample(scale_factor=2)),
            ('conv2d4', nn.Conv2d(96, 64, (3, 3), padding= 1)),
            ('bn4', nn.BatchNorm2d(64, momentum= 1)), 
            ('ac4', nn.ReLU(inplace=True))
        ]))

    # this part is not UNet, this is encoder decoder
    def forward(self, x):
        out = self.features(x)
        out = self.decode(out)
        return out
