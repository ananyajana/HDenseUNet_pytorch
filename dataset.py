import torch
import torch.utils.data as data
from PIL import Image
import h5py
import numpy as np

def get_len(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        return len(f['CT']['img'])

class LiTSDataset(data.Dataset):
    def __init__(self, hdf5_path, data_transform):
        super(LiTSDataset, self).__init__()
        self.hdf5_path = hdf5_path
        self.data_transform = data_transform
        self.len = get_len(self.hdf5_path)

    def __getitem__(self, index):
        hdf5_file = h5py.File(self.hdf5_path, "r")
        
        all_img_keys = list(hdf5_file['CT']['img'].keys())
        all_seg_keys = list(hdf5_file['CT']['seg'].keys())
        
        img = all_img_keys[index]
        seg = all_seg_keys[index]
        
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

