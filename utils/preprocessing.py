#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:10:41 2019

@author: liqinying
"""

import numpy as np
import pandas as pd
import torch
import os

import skimage.io as io

from cv2 import createThinPlateSplineShapeTransformer, DMatch, resize
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
  def __init__(self, root_dir, csv_file, transform=None):
    super(ImageLandmarksDataset,self).__init__()
    """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
    # TODO: Change back. This is just to speed up debugging.
    self.landmarks_frame = pd.read_csv(csv_file).head(100)
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
      sample = self.transform(sample)
    return sample
  
  def get_tensor(self):
    np_list=list()
    for i in range(len(self.landmarks_frame)):
      np_list.append(self.__getitem__(i)['image'])
    
    tensor=torch.stack(np_list)
    return tensor
  
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}
      
class CenterCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = (h - new_h)//2
        left = (w- new_w)//2

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class ColorJitter(object):
  def __init__(self, brightness:int, contrast:int, saturation:int, hue:float ):
    assert brightness>=0 and contrast>=0 and saturation>=0 and hue>=-0.5 and hue<=0.5, \
    "wrong parameter for colorjitter" 
    self.brightness=brightness
    self.contrast=contrast
    self.saturation=saturation
    self.hue=hue
  
  def __call__(self, sample):
    img, landmarks = sample['image'], sample['landmarks']
    img = img * (self.contrast/127+1) - self.contrast + self.brightness
    img = np.clip(img, 0, 255)
    img = img.astype(np.double)
    # (TODO): implement hue/sat.
    return {'image':img,
            'landmarks':landmarks}
  
class ThinPlateSpline(object):
  """
  used
  """
  def __init__(self, max_point:int, rand_point=True, distortion=5):
    assert max_point>=3, "the number of maximum control points should be bigger than 3"
    assert distortion>0, "no negative distortion"
    self.max_point=max_point
    self.rand_point=rand_point
    self.distortion=int(distortion)
  
  def __call__(self, sample):
    image, landmarks = sample['image'], sample['landmarks']
    h, w = image.shape[:2]
    assert self.distortion<min(int(h/5), int(w/5))

    tps = createThinPlateSplineShapeTransformer()
    corner1=[0,0]
    corner2=[h,0]
    corner3=[0,w]
    corner4=[h,w]
    #choose the rest of the grid points around the center to avoid distorting the background

    grid_upper=int(h/4)
    grid_lower=3*grid_upper
    grid_left=int(w/4)
    grid_right=3*grid_left

    if self.rand_point:
      n_points=np.random.randint(3, self.max_point)
    else:
      n_points=self.max_point

    tpoints=np.empty((n_points, 2), dtype=np.int32)
    spoints=np.empty((n_points, 2), dtype=np.int32)

    for i in range(0, n_points):
      x=np.random.randint(grid_upper, grid_lower)
      y=np.random.randint(grid_left, grid_right)
      d_y=np.random.randint(1, self.distortion)
      d_x=np.random.randint(1, self.distortion)
      tpoints[i]=np.asarray([x, y])
      spoints[i]=np.asarray([x+d_x, y+d_y])

    tpoints=np.concatenate([tpoints, [corner1, corner2, corner3, corner4]], axis=0)
    spoints=np.concatenate([spoints, [corner1, corner2, corner3, corner4]], axis=0) 

    spoints = spoints.reshape(1,-1,2)
    tpoints = tpoints.reshape(1,-1,2)
    landmarks = landmarks.reshape(1,-1,2)
    matches = list()

    for i in range(n_points+4):
      matches.append(DMatch(i,i,0))

    tps.estimateTransformation(tpoints,spoints,matches)
    out_img = tps.warpImage(image).astype(np.double)
    
    #CAUTION!!! landmarks not transformed!!!
    return {'image':out_img,
            'landmarks':landmarks}
  
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
