import os
import numpy as np
import PIL.Image
import scipy.io as sio
import torch
from torch.utils import data
import cv2
from utils.functions import adaptive_bins, get_bins_masks

class MyData(data.Dataset):
    """
       load data in a folder
    """
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])


    def __init__(self, root, transform=False):
        super(MyData, self).__init__()
        self.root = root

        self._transform = transform
        img_root = os.path.join(self.root, 'train_images')
        mask_root = os.path.join(self.root, 'train_masks')
        depth_root = os.path.join(self.root, 'train_depth')
        file_names = os.listdir(img_root)
        self.img_names = []
        self.mask_names = []
        self.depth_names = []
        for i, name in enumerate(file_names):
            if not name.endswith('.jpg'):
                continue
            ## training with 2 dataset
            # if len(name.split('_')[0]) ==4 :
            #     continue
            # print(name)
            self.mask_names.append(
                os.path.join(mask_root, name[:-4] + '.png')
            )

            self.img_names.append(
                os.path.join(img_root, name)
            )
            self.depth_names.append(
                os.path.join(depth_root, name[:-4] + '.png')
            )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        mask_file = self.mask_names[index]
        mask = PIL.Image.open(mask_file)
        mask = np.array(mask, dtype=np.int32)
        mask[mask != 0] = 1
        # load depth
        depth_file = self.depth_names[index]
        depth = PIL.Image.open(depth_file)
        depth = np.array(depth, dtype=np.uint8)
        # bins
        bins_mask = get_bins_masks(depth)

        if self._transform:
            return self.transform(img, mask, depth, bins_mask)
        else:
            return img, mask, depth, bins_mask

    def transform(self, img, mask, depth, bins_mask):
        img = img.astype(np.float64)/255.0
        img -= self.mean_rgb
        img /= self.std_rgb
        img = img.transpose(2, 0, 1)  # to verify
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()
        depth = depth.astype(np.float64) / 255.0
        depth = torch.from_numpy(depth).float()
        
        bins_mask=torch.from_numpy(bins_mask).float()
        h,w=depth.size()
        bins_depth = depth.view(1, h, w).repeat(3, 1, 1)
        bins_depth=bins_depth * bins_mask
        for i in range(3):
            bins_depth[i]=bins_depth[i]/bins_depth[i].max()
        c, h, w = img.size() 
        return img, mask, depth, bins_depth#
       
    



class MyTestData(data.Dataset):
    """
    load data in a folder
    """
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])

    def __init__(self, root, transform=False, use_bins=True):
        super(MyTestData, self).__init__()
        self.root = root
        self._transform = transform
        self._bins = use_bins

        img_root = os.path.join(self.root, 'test_images')
        depth_root = os.path.join(self.root, 'test_depth')
        file_names = os.listdir(img_root)
        self.img_names = []
        self.names = []
        self.depth_names = []

        for i, name in enumerate(file_names):
            if not name.endswith('.jpg'):
                continue
            self.img_names.append(
                os.path.join(img_root, name)
            )
            self.names.append(name[:-4])
            self.depth_names.append(
                os.path.join(depth_root, name[:-4] + '.png')
            )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file)
        img_size = img.size
        img = np.array(img, dtype=np.uint8)

        # load depth
        depth_file = self.depth_names[index]
        depth = PIL.Image.open(depth_file)
        depth = np.array(depth, dtype=np.uint8)

        bins_mask = get_bins_masks(depth)


        if self._transform:
            img, depth, bins_depth = self.transform(img, depth, bins_mask)
            return img, depth, bins_depth, self.names[index], img_size
        else:
            return img, depth, bins_mask, self.names[index], img_size



    def transform(self, img, depth, bins_mask):
        img = img.astype(np.float64)/255.0
        img -= self.mean_rgb
        img /= self.std_rgb
        img = img.transpose(2, 0, 1)  # to verify
        img = torch.from_numpy(img).float()
      
        depth = depth.astype(np.float64) / 255.0
        depth = torch.from_numpy(depth).float()
        
        bins_mask=torch.from_numpy(bins_mask).float()
        h,w=depth.size()
        bins_depth = depth.view(1, h, w).repeat(3, 1, 1)
        bins_depth=bins_depth * bins_mask
        for i in range(3):
            bins_depth[i]=bins_depth[i]/bins_depth[i].max()
        c, h, w = img.size() 
        return img, depth,bins_depth#

if __name__ == '__main__':
    root = "/data/wenhu/RGBD-SOD/SOD-RGBD/val/SIP"
    test_loader = torch.utils.data.DataLoader(MyTestData(root, transform=True),
                                   batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    for id, (data, depth, bins, img_name, img_size) in enumerate(test_loader):
        print(img_size)