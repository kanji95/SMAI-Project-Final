import os

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from PIL import Image

class ImageDataset(Dataset):
    
    def __init__(self, root, split='train', transform=None) -> None:
        
        self.root = root
        self.split = split
        self.transform = transform
        
        img_path = os.path.join(self.root, self.split)
        self.img_files = os.listdir(img_path)
        
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