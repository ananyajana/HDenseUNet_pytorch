"""Test ImageNet pretrained DenseNet"""
#from __future__ import print_function
import  numpy as np
import os

import sys
import torch
import torch.nn
from multiprocessing.dummy import Pool as ThreadPool
import random
from medpy.io import load
import numpy as np
import argparse

from PIL import Image
from denseunet import denseUnet
from skimage.transform import resize
from torchvision import transforms

import nibabel as nib
from nibabel.testing import data_path

parser = argparse.ArgumentParser(description='Keras 2d denseunet Training')
#  data folder
parser.add_argument('-data', type=str, default='data/', help='test images')
parser.add_argument('-save_path', type=str, default='Experiments/')
#  other paras
parser.add_argument('-b', type=int, default=40)
parser.add_argument('-input_size', type=int, default=224)
parser.add_argument('-model_weight', type=str, default='./model/densenet161_weights_tf.h5')
parser.add_argument('-input_cols', type=int, default=3)

#  data augment
parser.add_argument('-mean', type=int, default=48)
parser.add_argument('-thread_num', type=int, default=14)
args = parser.parse_args()

#file_path = '/research/cbim/medical/aj611/experiments/data/ILSVRC12/val2/val/n04264628/ILSVRC2012_val_00001672.JPEG'
file_path='../../data/ILSVRC12/val2/val/n04264628/ILSVRC2012_val_00001672.JPEG'
img = Image.open(file_path)
#trans = transforms.ToPILImage()
trans = transforms.ToTensor()
trans1 = transforms.Resize((224, 224))
trans2 = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
im = trans2(trans(trans1(img)))
print(im.size())
im = im.unsqueeze(0)
print(im.size())

model = denseUnet(reduction=0.5)
out = model(im)
print(out)
