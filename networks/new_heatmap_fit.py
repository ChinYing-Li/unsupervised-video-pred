#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 08:25:59 2019

@author: liqinying
"""
import torch
import torch.nn as nn 
import numpy as np

def main():
    b=3
    n_heat=2
    crop=5
    x=torch.randint(2,5, (b, n_heat, crop,crop))
    #print(x[0])
    xcoor, ycoor=np.indices(x.shape[-2:])
    xcoords=torch.from_numpy(xcoor).float()
    ycoords=torch.from_numpy(ycoor).float()
    total=torch.sum(torch.sum(x, dim=-1),dim=-1)
    area=torch.empty(1, dtype=torch.float)
    area[0]=crop*crop
    
    xcoords=xcoords.unsqueeze(0).unsqueeze(0)
    ycoords=ycoords.unsqueeze(0).unsqueeze(0)
    xcoords=xcoords.repeat(b, n_heat, 1, 1)
    ycoords=ycoords.repeat(b, n_heat, 1, 1)
    #print(xcoords[0])
    xcenter=torch.sum(torch.sum(torch.mul(x, xcoords), dim=-1),dim=-1)/total
    ycenter=torch.sum(torch.sum(torch.mul(x, ycoords), dim=-1),dim=-1)/total
    xcenter=xcenter.unsqueeze(-1).unsqueeze(-1)
    ycenter=ycenter.unsqueeze(-1).unsqueeze(-1)
    xcenter=xcenter.repeat(1, 1, crop, crop)
    ycenter=ycenter.repeat(1, 1, crop, crop)
    print(xcenter.shape)
    varx=torch.sum(torch.sum(torch.mul(xcoords-xcenter, xcoords-xcenter),dim=-1),dim=-1)
    vary=torch.sum(torch.sum(torch.mul(ycoords-ycenter, ycoords-ycenter),dim=-1),dim=-1)
    cov_xy=torch.sum(torch.sum(torch.mul(xcoords-xcenter, ycoords-ycenter),dim=-1),dim=-1)
    
    sig_buf=torch.stack((varx, cov_xy), dim=-1)
    sigma=torch.stack((cov_xy, vary), dim=-1)
    sigma=torch.stack((sig_buf, sigma), dim=-2)
    #print(sigma[0])
    
    coords_normed=torch.stack((xcoords,ycoords), dim=-1)-torch.stack((torch.ones(x.shape, dtype=torch.float)*xcenter,\
                                      torch.ones(x.shape, dtype=torch.float)*ycenter),dim=-1)
    #print(coords_normed[0])
    print(torch.stack((xcoords,ycoords), dim=-1))
    #coords_normed=torch.transpose(coords_normed, 0, 1)
    #coords_normed=torch.transpose(coords_normed, 1,2)
    sigma_inv=torch.inverse(sigma)
    sigma_inv=sigma_inv.unsqueeze(-3)
    sigma_inv=sigma_inv.unsqueeze(-3)
    sigma_inv=sigma_inv.repeat(1,1,crop,crop, 1,1)
    print(sigma_inv.shape)
    print(coords_normed.shape)
    out=1./(1+torch.matmul(torch.matmul(coords_normed.unsqueeze(-2), sigma_inv),coords_normed.unsqueeze(-1)))
    print(out.shape)

if __name__=="__main__":
    main()