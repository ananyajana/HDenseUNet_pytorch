"""
This file builds patient wise datasets i.e. all the images for a patient are under
a single idx i. This is suitable for our final training.
For UNet, we do not need patient wse training yet, as we are not performing volumetric
training.
"""

import os
import h5py
import math
import numpy as np
from skimage import io
from tqdm import tqdm
import argparse
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tempfile import TemporaryFile


parser = argparse.ArgumentParser(description='')
parser.add_argument('--output_path', type=str, default='./data_all_no_empty_mask', help='custom data path')
parser.add_argument('--src', type=str, default='', help='custom data path')
parser.add_argument('--mode', type=str, default='Train', help='train or test mode')
parser.add_argument('--h5_filename', type=str, default='lits_train.h5', help='combined h5 filename')
parser.add_argument('--max_hu_val', type=int, default=250)
parser.add_argument('--min_hu_val', type=int, default=-200)
parser.add_argument('--train_test_split', type=float, default=0.8, help='specify the train test ratio')
parser.add_argument('--seed', type=int, default=23, help='seed for random choice')
parser.add_argument('--remove_empty_mask', type=int, default=0, help='remove empty mask? 1(yes), 0(no)')
#---------- data_two changes start-----#
#parser.add_argument('--output_path', type=str, default='./data_two_no_empty_mask', help='custom data path')
#---------- data_two changes end-----#
args = parser.parse_args()


src_path = args.src
max_window_value = args.max_hu_val
min_window_value = args.min_hu_val
train_test_ratio = args.train_test_split
train_str = args.mode
save_path = args.output_path
h5_filename = args.h5_filename
seed = args.seed

max_window_value = 250
min_window_value = -200
src_path = '../../data/LiTS/'


# if the mode is 'Train' then create the h5 file out of train volumes
# else if it is in Test mode then create the h5 file out of test 
# volumes(which do not have corresponding segmentation)

if train_str in 'Train':
    num_patients = 131
elif train_str in 'Test':
    num_patients = 70

# do the train-test split our of the 131 volumes
train_set_len = math.floor(num_patients * train_test_ratio)
# randomly pick train_set_len elements from the set of all indicesnp.

#---------- data_two changes start-----#
#num_patients = 2
#train_set_len = math.floor(num_patients * train_test_ratio)
#---------- data_two changes end-----#

np.random.seed(seed)
train_set = np.random.choice(num_patients, train_set_len, replace=False)
# take the remainng patients in test list
test_set = list(set(np.arange(num_patients)).difference(train_set))

#---------- data_two changes start-----#
print('train_set', train_set)
print('test_set', test_set)
#---------- data_two changes end-----#

os.makedirs(save_path, exist_ok=True)
train_h5_filepath = save_path + '/' + 'train.h5'
test_h5_filepath = save_path + '/' + 'test.h5'

train_dataset = h5py.File(train_h5_filepath, 'w')
test_dataset = h5py.File(test_h5_filepath, 'w')

def create_h5(dataset, patients=None):
    cnt = 0 # cnt of total number of images(= segmentations) in the Train set including data from
    # all the volumes
    dataset.create_group('CT')
    for k in range(len(patients)):
        i = patients[k]
        print('in the loop')
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
            '''
            img_slice = src_img[idx, :, :]
            img_slice = img_slice.squeeze()
            dataset.create_dataset('CT/img/{:d}'.format(cnt), data=img_slice)
            '''
            #dataset.create_dataset('img/{:d}'.format(cnt), data=img_slice)
            if train_str in 'Train':
                seg_slice = seg_img[idx, :, :]
                # if the segmentation contains anything other than background
                if len(np.unique(seg_slice)) > 1:
                    seg_slice = seg_slice.squeeze()
                    dataset.create_dataset('CT/seg/{:d}'.format(cnt), data=seg_slice)
                    #dataset.create_dataset('seg/{:d}'.format(cnt), data=seg_slice)
                    
                    img_slice = src_img[idx, :, :]
                    img_slice = img_slice.squeeze()
                    dataset.create_dataset('CT/img/{:d}'.format(cnt), data=img_slice)
                        
                    cnt += 1    #increment cnt as a new slice has been added to dataset
    dataset.close()

create_h5(train_dataset, train_set)
create_h5(test_dataset, test_set)
