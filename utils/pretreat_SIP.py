import os
import numpy as np
import PIL.Image
import scipy.io as sio
import torch
from torch.utils import data
import cv2
from utils.functions import adaptive_bins, get_bins_masks

root = "/data/wenhu/RGBD-SOD/SOD-RGBD/val/raw_SIP"

img_root = os.path.join(root, 'test_images')
depth_root = os.path.join(root, 'test_depth')
gt_root = os.path.join(root,'test_masks')
names=[]
img_names=[]
depth_names=[]
gt_names=[]

file_names = os.listdir(img_root)

for i, name in enumerate(file_names):
    if not name.endswith('.jpg'):
        continue
    names.append(name[:-4])
    img_names.append(
        os.path.join(img_root, name)
    )
    
    depth_names.append(
        os.path.join(depth_root, name[:-4] + '.png')
    )

    gt_names.append(
        os.path.join(gt_root, name[:-4] + '.png')
    )

new_root = "/data/wenhu/RGBD-SOD/SOD-RGBD/val/SIP"
new_img_root = os.path.join(new_root, 'test_images')
new_depth_root = os.path.join(new_root, 'test_depth')
new_gt_root = os.path.join(new_root,'test_masks')

if not os.path.exists(new_depth_root):
    os.mkdir(new_depth_root)
if not os.path.exists(new_img_root):
    os.mkdir(new_img_root)
if not os.path.exists(new_gt_root):
    os.mkdir(new_gt_root)

# i=0
# print(gt_names[0])
# img = np.array(PIL.Image.open(img_names[i]))
# depth = np.array(PIL.Image.open(depth_names[i]))
# gt = np.array(PIL.Image.open(gt_names[i]))
# print(img.shape, depth.shape, gt.shape)




for i in range(len(img_names)):
    img = cv2.imread(img_names[i])
    depth = cv2.imread(depth_names[i])
    gt = cv2.imread(gt_names[i])

    img = cv2.resize(img, (512,512), interpolation = cv2.INTER_LINEAR)
    depth = cv2.resize(depth, (512,512), interpolation = cv2.INTER_LINEAR)[:,:,0]
    gt = cv2.resize(gt, (512,512), interpolation = cv2.INTER_LINEAR)[:,:,0]

    cv2.imwrite( os.path.join(new_img_root, names[i]+ '.jpg'), img )
    cv2.imwrite( os.path.join(new_depth_root, names[i]+ '.png'), depth )
    cv2.imwrite( os.path.join(new_gt_root, names[i]+ '.png'), gt )

    # if i>10:
    #     break
