import numpy as np
import matplotlib.pyplot as plt
import torch
# from scipy.misc import imresize
from PIL import Image
import os
import cv2

def adaptive_bins( hist,threshold):
    new = hist.copy()
    peak=hist.max()
    peak_depth=np.where(hist==peak)[0]
    delta_hist=np.diff(hist,n=1,axis=0)
    #print(peak,peak_depth,peak_depth.shape)
    left=peak_depth
    right=peak_depth
    i = np.array([peak_depth[0]])
    while(1):
        new[[i]]=0
        if (i>=254):
            right=np.array([254])
            break
        if (delta_hist[i]<0):
            i=i+1
        elif (hist[i]<=threshold*peak):
            right = i
            break
        else:
            i=i+1
    i = np.array([peak_depth[0]-1])
    while(1):
        new[[i+1]]=0
        if (i<=0):
            left=np.array([0])
            break
        if (delta_hist[i]>0):
            i=i-1
        elif (hist[i]<=threshold*peak):
            left = i+1
            break
        else:
            i=i-1              
    #print(peak,peak_depth,left[0],right[0])
    return [new,left[0],right[0]]

def get_bins_masks( depth):
        mask_list=[]
        hist = cv2.calcHist([depth],[0],None,[256],[0,255])  
        
        hist1,left1,right1=adaptive_bins(hist,0.7)
        mask1 = (depth>left1-0.2*(right1-left1)) * (depth<=right1+0.2*(right1-left1))
        #mask1 = (depth>left1) * (depth<=right1)
        
        mask_list.append(mask1)
        
        hist2,left2,right2=adaptive_bins(hist1,0.2)
        mask2 = (depth>left2-0.2*(right2-left2)) * (depth<=right2+0.2*(right2-left2))
        #mask2 = (depth>left2) * (depth<=right2)
    
        mask_list.append(mask2)

        mask3_1 =(depth>left1) * (depth<=right1)
        mask3_2 =(depth>left2) * (depth<=right2)
        mask3=(~mask3_2)*(~mask3_1)
        
        mask_list.append(mask3)
        mask_bins = np.stack(mask_list,axis=0)

        return mask_bins


def create_exp_dir(path, scripts_to_save=None):
  import time
  time.sleep(2)
  if not os.path.exists(path):
    os.makedirs(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.makedirs(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)



def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for v in model.parameters())/1e6

  

def imsave(file_name, img, img_size):
    """
    save a torch tensor as an image
    :param file_name: 'image/folder/image_name'
    :param img: 3*h*w torch tensor
    :return: nothing
    """
    assert(type(img) == torch.FloatTensor,
           'img must be a torch.FloatTensor')
    ndim = len(img.size())
    assert(ndim == 2 or ndim == 3,
           'img must be a 2 or 3 dimensional tensor')

    img = img.numpy()
    
    img = np.array(Image.fromarray(img).resize((img_size[1][0], img_size[0][0]), Image.NEAREST))
#     img = imresize(img, [img_size[1][0], img_size[0][0]], interp='nearest')
    if ndim == 3:
        plt.imsave(file_name, np.transpose(img, (1, 2, 0)))
    else:
        plt.imsave(file_name, img, cmap='gray')

def load_pretrain(path, state_dict, name):
    state = torch.load(path)
    if 'state_dict' in state:
        state = state['state_dict']
    name = "module."+name
    length = len(name)
    for k, v in state.items():
        if k[:length] == name:
            if k[length:] in state_dict.keys():
                state_dict[k[length:]] = v
                # print(k[length:])
            else:
                print("pass keys: ",k[7:])

    return state_dict
