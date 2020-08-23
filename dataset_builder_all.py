import cv2
import os
import h5py
import numpy as np
from skimage import io
from tqdm import tqdm
import argparse
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tempfile import TemporaryFile


parser = argparse.ArgumentParser(description='')
parser.add_argument('--output_path', type=str, default='./data', help='custom data path')
parser.add_argument('--src', type=str, default='', help='custom data path')
parser.add_argument('--mode', type=str, default='Train', help='train or test mode')
parser.add_argument('--h5_filename', type=str, default='lits_train.h5', help='combined h5 filename')
parser.add_argument('--max_hu_val', type=int, default=250)
parser.add_argument('--min_hu_val', type=int, default=-200)
args = parser.parse_args()


src_path = args.src
max_window_value = args.max_hu_val
min_window_value = args.min_hu_val

max_window_value = 250
min_window_value = -200
src_path = '/dresden/users/aj611/experiments/data/LiTS/'

save_path = args.output_path
h5_filename = args.h5_filename


# if the mode is 'Train' then create the h5 file out of train volumes
# else if it is in test mode then create the h5 file out of test 
# volumes(which do not have corresponding segmentation)
train_str = args.mode
os.makedirs(save_path, exist_ok=True)
h5_filepath = save_path + '/' + h5_filename
dataset = h5py.File(h5_filepath, 'w')

if train_str in 'Train':
    num_patients = 131
else:
    num_patients = 70
num_patients = 3

for i in range(num_patients):
    print('in the loop')
    dataset.create_group('{:d}/{:s}/img'.format(i, train_str))
    # get the volume 
    src = sitk.ReadImage(src_path + train_str + '/volume-' + str(i) + '.nii')
    # get the numpy array from the volumes
    src_img = sitk.GetArrayFromImage(src)
    print(src_img)
    # perform Hounsfield windowing on source images(only)
    src_img[src_img > max_window_value] = max_window_value
    src_img[src_img < min_window_value] = min_window_value

    # similar processing on segmentation nii file  
    if train_str in 'Train':
        dataset.create_group('{:d}/{:s}/seg'.format(i, train_str))
        seg = sitk.ReadImage(src_path + train_str + '/segmentation-' + str(i) + '.nii')
        seg_img = sitk.GetArrayFromImage(seg)
        # the labels legend: 0: background, 1: liver, 2: liver tumor
        # since we are only interested in the liver we convert
        # tumor labels into liver label as the tumor is part of the liver region
        # we do not need the tumor segmentation and hence
        # converting the tumor labels to liver labels
        seg_img[seg_img == 2] = 1  
    
    # get the src image shape to see how many slices(=depth or #channels) are present. 
    # extract the slices as these need to go inside the volume individually
    d, h, w = src_img.shape
    print(d)
    for idx in range(d):
        img_slice = src_img[idx, :, :]
        img_slice = img_slice.squeeze()
        dataset.create_dataset('{:d}/{:s}/img/{:d}'.format(i, train_str, idx), data=img_slice)
        if train_str in 'Train':
            seg_slice = seg_img[idx, :, :]
            seg_slice = seg_slice.squeeze()
            dataset.create_dataset('{:d}/{:s}/seg/{:d}'.format(i, train_str, idx), data=seg_slice)

dataset.close()

def split_train_test(h5_filepath, save_dir):
    import random

    os.makedirs(save_dir, exist_ok=True)
    h5_file = h5py.File(h5_filepath, 'r')

    keys = list(h5_file.keys())
    indices = list(range(len(keys)))
    random.seed(23)
    random.shuffle(keys)

    N_fold = 3
    l = int(np.ceil(len(keys) / N_fold))

    # cross validation: fold i
    for fold in range(N_fold):
        print('Fold {:d}'.format(fold+1))
        end = l*(fold+1) if l*(fold+1) <= len(indices) else len(indices)
        test_indices = list(range(l*fold, end))
        train_indices = [idx for idx in indices if idx not in test_indices]
        print('test indices')
        print(sorted(test_indices))
        print('train indices')
        print(sorted(train_indices))

        train_file = h5py.File('{:s}/train{:d}.h5'.format(save_dir, fold+1), 'w')
        test_file = h5py.File('{:s}/test{:d}.h5'.format(save_dir, fold+1), 'w')
        for idx in train_indices:
            h5_file.copy(keys[idx], train_file)
        for idx in test_indices:
            h5_file.copy(keys[idx], test_file)
        train_file.close()
        test_file.close()

    h5_file.close()

split_train_test(h5_filepath, save_path)
