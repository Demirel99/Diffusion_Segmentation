# file: dataset_isic_points.py
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import random

class ISICPointDataset(Dataset):
    def __init__(self, img_dir, mask_dir, file_list, image_size=224, augment=True, img_ext='.jpg', mask_suffix='_segmentation.png'):
        """
        Dataset for ISIC images and their corresponding sparse point masks.
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment
        self.img_ext = img_ext
        self.mask_suffix = mask_suffix
        
        self.img_files = file_list
        
        self.transform = transforms.Compose([
            # Images are assumed to be pre-resized by the preparation script
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
            
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_name = self.img_files[index]
        base_name = img_name.replace(self.img_ext, '')
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_name = f"{base_name}{self.mask_suffix}"
        mask_path = os.path.join(self.mask_dir, mask_name)

        cond_image = Image.open(img_path).convert('RGB')
        point_mask = Image.open(mask_path).convert('L')

        if self.augment and random.random() > 0.5:
            cond_image = cond_image.transpose(Image.FLIP_LEFT_RIGHT)
            point_mask = point_mask.transpose(Image.FLIP_LEFT_RIGHT)

        cond_image_tensor = self.transform(cond_image)
        
        # This is the sparse point mask, which will be used as x_start
        dot_map_tensor = self.mask_transform(point_mask)
        dot_map_tensor = (dot_map_tensor > 0.5).float()
        
        return cond_image_tensor, dot_map_tensor