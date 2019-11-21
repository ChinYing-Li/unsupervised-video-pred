#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:48:30 2019

@author: liqinying
"""

import numpy as np
import pandas as pd
import torch
import os

import skimage.io as io
from torch.utils.data import Dataset

"""
https://github.com/jbohnslav/opencv_transforms/blob/master/opencv_transforms.py

^^^^^^^^^^^^^^WE SHOULD USE THIS^^^^^^^^^^^^^^^^
"""

#define class CelebAdataset !??
"""
path="/"
data_list=list()
for filename in listdir(path):
  img=image.imread(path+filename)
  data_list.append(img)
  
data_list=asarray(data_list)
data_list=torch.from_numpy(data_list)
"""

class ImageLandmarksDataset(Dataset):
  def __init__(self, root_dir, ann_file=None, transform=None):
    super(ImageLandmarksDataset,self).__init__()
    """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
    # TODO: Change back. This is just to speed up debugging.
    self.landmarks_frame = pd.read_csv(ann_file).head(100)
    self.root_dir = root_dir
    self.transform = transform
        
  def __len__(self):
    return len(self.landmarks_frame)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
    # print(img_name)
    image = io.imread(img_name)
    landmarks = self.landmarks_frame.iloc[idx, 1:]
    landmarks = np.array([landmarks])
    landmarks = landmarks.astype('float').reshape(-1, 2)
    sample = {'image': image, 'landmarks': landmarks}

    if self.transform:
      sample['image'] = self.transform(sample['image'])
    return sample
  
  def get_tensor(self):
    np_list=list()
    for i in range(len(self.landmarks_frame)):
      np_list.append(self.__getitem__(i)['image'])
    
    tensor=torch.stack(np_list)
    return tensor