import torch
import torch.utils.data as data
from PIL import Image
import h5py
import numpy as np
from random import randint
from skimage.transform import resize

from utils import normalization

def get_len(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        return len(f['CT']['img'])

class LiTSDataset(data.Dataset):
    def __init__(self, hdf5_path, data_transform, random_mirror=False, scale=False):
        super(LiTSDataset, self).__init__()
        self.hdf5_path = hdf5_path
        print('hdf5 path:', self.hdf5_path)
        self.data_transform = data_transform
        self.len = get_len(self.hdf5_path)
        self.random_mirror = random_mirror
        self.scale = scale
        self.hdf5_file = h5py.File(self.hdf5_path, "r")
        self.all_img_keys = list(self.hdf5_file['CT']['img'].keys())
        self.all_seg_keys = list(self.hdf5_file['CT']['seg'].keys())
        
        # some debug messages
        print('all_img_keys: ', self.all_img_keys)
        print('all_seg_keys: ', self.all_seg_keys)
        print('mirror: ', self.random_mirror)
        print('scale: ', scale)
    def __getitem__(self, index):
        img_key = self.all_img_keys[index]
        seg_key = self.all_seg_keys[index]
        
        img = self.hdf5_file['CT']['img'][img_key][()]
        seg = self.hdf5_file['CT']['seg'][seg_key][()]
        
        if self.random_mirror is True:
            flip_num = randint(0, 7)
            print('random mirror')
            if flip_num == 1:
                img = np.flipud(img)
                seg = np.flipud(seg)
            elif flip_num == 2:
                img = np.fliplr(img)
                seg = np.fliplr(seg)
            elif flip_num == 3:
                img = np.rot90(img, k=1, axes=(1, 0))
                seg = np.rot90(seg, k=1, axes=(1, 0))
            elif flip_num == 4:
                img = np.rot90(img, k=3, axes=(1, 0))
                seg = np.rot90(seg, k=3, axes=(1, 0))
            elif flip_num == 5:
                img = np.fliplr(img)
                seg = np.fliplr(seg)
                img = np.rot90(img, k=1, axes=(1, 0))
                seg = np.rot90(seg, k=1, axes=(1, 0))
            elif flip_num == 6:
                img = np.fliplr(img)
                seg = np.fliplr(seg)
                img = np.rot90(img, k=3, axes=(1, 0))
                seg = np.rot90(seg, k=3, axes=(1, 0))
            elif flip_num == 7:
                img = np.flipud(img)
                seg = np.flipud(seg)
                img = np.fliplr(img)
                seg = np.fliplr(seg)
                
        if self.scale is True:
            #  randomly scale
            scale = np.random.uniform(0.8,1.2)
            h,w = img.shape
            h = int(h * scale)
            w = int(w * scale)
            img = resize(img, (h, w), order=3, mode='edge', cval=0, clip=True, preserve_range=True)
            seg = resize(seg, (h, w), order=3, mode='edge', cval=0, clip=True, preserve_range=True)
        
        img = normalization(img, max=1, min=0)
        
        img = Image.fromarray(img.astype('uint8'), 'L')
        seg = Image.fromarray(seg.astype('uint8'), 'L')
        
        img_tensor = []
        seg_tensor = []
        
        img_tensor.append(self.data_transform(img).unsqueeze(0))
        seg_tensor.append(self.data_transform(seg).unsqueeze(0))
        

        return torch.cat(img_tensor, dim=0), \
            torch.cat(seg_tensor, dim=0)

    def __len__(self):
        return self.len

