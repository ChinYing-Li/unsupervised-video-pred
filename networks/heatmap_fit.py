#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:54:36 2019

@author: liqinying
"""

import torch
import torch.nn as nn
import numpy as np

class heatmap_fit(nn.Module):
  def __init__(self, n_heatmaps=None, batch_size=4):
    super(heatmap_fit, self).__init__()
    self.n_heatmaps=None
    if n_heatmaps:
      self.n_heatmaps=n_heatmaps
    
    self.batch_size=batch_size

    #self.mu_x=torch.tensor([0.], requires_grad=True)
    #self.mu_y=torch.tensor([0.], requires_grad=True)

  def forward(self, x):
    
    #input tensor shpae K x H x W
    if self.n_heatmaps==None:
      self.n_heatmaps=x.shape[1]

    xcoords, ycoords=np.indices(x[0][0].shape)
    xcoords=torch.from_numpy(xcoords).float()
    ycoords=torch.from_numpy(ycoords).float()
    out=torch.empty(x.shape, dtype=torch.float, requires_grad=True)
    coords=torch.stack([xcoords, ycoords], dim=0)
    """for i in self.inp.shape[0]:
      for j in self.inp.shape[1]:"""
    
    #use view or expand to 
    for b in range(self.batch_size):
      for i in range(self.n_heatmaps):
        sigma, mu_x, mu_y=self.calc_sigma(x[b][i])
        sigma_inv=torch.inverse(sigma)
        coords_normed=coords-torch.stack([torch.ones(xcoords.shape, dtype=torch.float)*mu_x,\
                                      torch.ones(ycoords.shape, dtype=torch.float)*mu_y])
    
        coords_normed=torch.transpose(coords_normed, 0, 1)
        coords_normed=torch.transpose(coords_normed, 1,2)
        print("coords_normed")
        #print(coords_normed)
        for m in range(coords_normed.shape[0]):
          for n in range(coords_normed.shape[1]):
            out[b][i][m][n]=1./(1.+torch.matmul(torch.matmul(coords_normed[m][n], sigma_inv), torch.unsqueeze(coords_normed[m][n],-1)))

    return out
    """
    helper functions
    """
  
  def cov(self, m, y=None):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    covariance = 1 / (x.size(1) - 1) * x.mm(x.t())
    return covariance

  def calc_sigma(self, raw):
    inp=raw
    ycoords, xcoords=np.indices(raw.shape)
    xcoords=torch.from_numpy(xcoords).float()
    ycoords=torch.from_numpy(ycoords).float()
    #raw=torch.from_numpy(raw)
    total=torch.sum(raw)
    shape=raw.shape
    area=torch.tensor([shape[0]*shape[1]], dtype=torch.float)
    sigma=torch.empty(2, 2, dtype=torch.float, requires_grad=True)

    xcenter=torch.tensor([torch.sum(torch.mul(inp, xcoords))/total], dtype=torch.float, requires_grad=True)
    
    ycenter=torch.tensor([torch.sum(torch.mul(inp, ycoords.float()))/total], dtype=torch.float, requires_grad=True)
    
    var_x=torch.tensor(torch.sum(torch.mul(xcoords-xcenter, xcoords-xcenter))/area, dtype=torch.float, requires_grad=True)
    var_y=torch.tensor(torch.sum(torch.mul(ycoords-ycenter, ycoords-ycenter))/area, dtype=torch.float, requires_grad=True)
    cov_xy=torch.tensor(torch.sum(torch.mul(xcoords-xcenter, ycoords-ycenter))/area, dtype=torch.float, requires_grad=True)
    """
    x_flat=raw.view(-1)
    
    trans=torch.transpose(raw, 0, 1).contiguous()
    imshow(torch.Tensor.numpy(trans))
    y_flat=trans.view(-1)
    xy_flat=torch.empty([2,x_flat.shape[0]], dtype=torch.float)
    xy_flat[0]=x_flat
    xy_flat[1]=y_flat
    xy_cov=self.cov(xy_flat)
    #print(xy_cov)"""
    sigma[0][0]=var_x
    sigma[1][1]=var_y
    sigma[0][1]=cov_xy
    sigma[1][0]=sigma[0][1]
    print(sigma)
    return sigma, xcenter, ycenter