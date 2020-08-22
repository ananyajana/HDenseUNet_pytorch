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


def weight_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 0.02)
            nn.init.zeros_(m.bias)

class denseUnet(nn.Module):
    def __init__(self, growth_rate=48, block_config=(6, 12, 36, 24), num_init_features=96, drop_rate=0, weight_decay=1e-4, num_classes=1000, reduction=0.0):
        super(denseUnet, self).__init__()
        nb_filter = num_init_features
        eps = 1.1e-5
        compression = 1 - reduction
        # initial convolution
        self.conv0_ = nn.Conv2d(3, nb_filter, kernel_size=7, stride=2,
                                padding=3, bias=False)
        self.norm0_ = nn.BatchNorm2d(nb_filter, eps= eps)
        self.scale0_ = Scale(nb_filter)
        self.relu0_ = nn.ReLU(inplace=True)
        self.pool0_ = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # dense block followed by transition
        self.block1 = dense_block(num_layeri[0], nb_filter, growth_rate, drop_rate)
        nb_filter += num_layer[0] * growth_rate
        self.trans1 = _Transition(nb_filter, nb_filter * compression)
        nb_filter = nb_filter * compression

        self.block2 = dense_block(num_layeri[1], nb_filter, growth_rate, drop_rate)
        nb_filter += num_layer[1] * growth_rate
        self.trans2 = _Transition(nb_filter, nb_filter * compression)
        nb_filter = nb_filter * compression

        self.block3 = dense_block(num_layeri[2], nb_filter, growth_rate, drop_rate)
        nb_filter += num_layer[2] * growth_rate
        self.trans3 = _Transition(nb_filter, nb_filter * compression)
        nb_filter = nb_filter * compression

        self.block4 = dense_block(num_layeri[3], nb_filter, growth_rate, drop_rate)
        nb_filter += num_layer[3] * growth_rate

        self.norm5 = nn.BatchNorm2d(nb_filter, eps= eps, momentum= 1)
        self.scale5 = Scale(nb_filter)
        self.relu5 = nn.ReLU(inplace= True)


        # the other half of the UNet
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(nb_filter, 2208, (1, 1), padding= 1)

        self.conv0 = nn.Conv2d(2208, 768, (3, 3), padding= 1)
        self.bn0 =  nn.BatchNorm2d(768, momentum= 1)
        self.ac0 = nn.ReLU(inplace=True)
        
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(768, 384, (3, 3), padding= 1)
        self.bn1 = nn.BatchNorm2d(384, momentum= 1)
        self.ac1 = nn.ReLU(inplace=True)

        self.up2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(384, 96, (3, 3), padding= 1)
        self.bn2 = nn.BatchNorm2d(96, momentum= 1)
        self.ac2 = nn.ReLU(inplace=True)

        self.up3 = nn.Upsample(scale_factor=2)
        self.conv3 = nn.Conv2d(96, 96, (3, 3), padding= 1)
        self.bn3 = nn.BatchNorm2d(96, momentum= 1)
        self.ac4 = nn.ReLU(inplace=True)

        self.up4 = nn.Upsample(scale_factor=2)
        self.conv4 = nn.Conv2d(96, 64, (3, 3), padding= 1)
        self.dropout = F.Dropout(p=0.3)
        self.bn4 = nn.BatchNorm2d(64, momentum= 1)
        self.ac4 = nn.ReLU(inplace=True)

        # last convolution
        self.conv5 = nn.Conv2d(64, 3, kernel_size=1)

        weight_init(self)

    # this part is not UNet, this is encoder decoder
    def forward(self, x):
        box = []
        out = self.ac0_(self.scale0_(self.norm0_(self.conv0_(x))))
        box.append(out)
        out = self.pool0(out)
        
        out = self.block1(out)
        box.append(out)
        out = self.trans1(out)
        
        out = self.block2(out)
        box.append(out)
        out = self.trans2(out)
        
        out = self.block3(out)
        box.append(out)
        out = self.trans3(out)
        
        out = self.block4(out)

        out = self.ac5(self.scale5(self.bn5(out)))
        box.append(out)

        up0 = self.up(out)
        line0 = self.conv(box[3])
        up0_sum = add([line0, up0])
        out = self.ac0(self.bn0(self.conv0(up0_sum)))

        up1 = self.up1(out)
        up1_sum = add([box[2], up1])
        out = self.ac1(self.bn1(self.conv1(up1_sum)))


        up2 = self.up2(out)
        up2_sum = add([box[1], up2])
        out = self.ac2(self.bn2(self.conv2(up2_sum)))

        up3 = self.up3(out)
        up3_sum = add([box[0], up3])
        out = self.ac3(self.bn3(self.conv3(up3_sum)))

        up4 = self.up3(out)
        out = self.ac4(self.bn4(self.dropout(self.conv4(up4))))

        out = self.conv5(out)

        return out
