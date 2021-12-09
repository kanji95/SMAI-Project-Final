import os

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from glob import glob
from PIL import Image

class Urban100(Dataset):
    def __init__(self, root, transform=None) -> None:
        self.root = root
        
        self.transform = transform
        img_path = os.path.join(self.root, 'rgb')
        self.img_files = glob(img_path + "/*.jpg")
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        img_file = self.img_files[index]
        img = Image.open(img_file)
        
        if self.transform:
            img = self.transform(img)
        else:
            img = ToTensor()(img)
            
        return img

class BSDDataset(Dataset):
    
    def __init__(self, root, split='train', transform=None) -> None:
        
        self.root = root
        self.split = split
        self.transform = transform
        
        img_path = os.path.join(self.root, self.split)
        self.img_files = glob(img_path + "/*.jpg")
        
        if self.split == 'train':
            test_path = os.path.join(self.root, 'test')
            self.img_files += glob(test_path + "/*.jpg")
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        img_file = self.img_files[index]
        img = Image.open(img_file)
        
        if self.transform:
            img = self.transform(img)
        else:
            img = ToTensor()(img)
            
        return img
