#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 20:18:57 2020

@author: aj611
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

def BCEWithLogits_2ddense(pred=None, target=None):
    # pass the predictions through a sigmoid layer
    pred_f = F.sigmoid(pred)
    # clamping the weights of the tensor
    pred_f = torch.clamp(pred_f, min=1e-10, max=1.0)
    # take log of the predicted values
    loss = -(pred_f.log()*target + (1-target)*(1-pred_f).log()).mean()
    return loss


def dice_loss(pred, target, smooth = 1.):
    print('Calculating Dice Loss')
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()


def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)
    m2 = target.view(num, -1)

    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) /(m1.sum() + m2.sum() + smooth)
