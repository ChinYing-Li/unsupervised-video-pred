#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 02:20:45 2019

@author: liqinying
"""

from .networks import heatmap_fit, Pose_Enc, App_Enc, MaskNet, BGNet, FG_Dec, Vgg19
from .utils.nn_utils import hadamard, invert

#from networks.subnetworks import Pose_Enc, App_Enc, MaskNet, BGNet, FG_Dec
#from networks.gaussian_fitting import Gaussian_Fit
#from networks.vgg import Vgg19

#from utils.nn_utils import hadamard

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

###
#import utils.preprocessing as img_dt # where class ImageLandmarksDataset is defined
###
from torchvision import transforms

import os
import time

class PipelineNet(nn.Module):
  def __init__(self, args):
    super(PipelineNet,self).__init__()
    
    self.n_heatmaps = args.n_heatmaps
    self.n_appearance = args.n_appearance
    self.norm = 'Batch'

    # For SPADE. what the hell am I trying to do?
    self.semantic_nc = args.n_heatmaps#???

    self.PoseEncoder=Pose_Enc(3, self.n_heatmaps, norm=self.norm)
    self.MaskNet=MaskNet(self.n_heatmaps, norm=self.norm)
    self.App_Encoder=App_Enc(args, norm=self.norm)
    self.FGDecoder=FG_Dec(args)
    self.BG_Net=BGNet()
    self.heatmap_fit=heatmap_fit(n_heatmaps=self.n_heatmaps, batch_size=args.batch_size)
    self.hadamard=hadamard()
    self.inv=invert()
    #I think we don't have to define hadamard and invert as members...

  def forward(self, x1, x2):
    img_cj=x1
    img_tps=x2

    x_cj=self.PoseEncoder(img_cj)
    x_tps=self.PoseEncoder(img_tps)
    
    gaussian_cj=self.heatmap_fit(x_cj)
    gaussian_tps=self.heatmap_fit(x_tps)
    
    x_app=self.App_Encoder(img_tps, map1=x_tps, map2=gaussian_cj)
    x_foreground=self.FGDecoder(gaussian_cj ,appearance_enco=x_app)
   
    x_cj=self.MaskNet(gaussian_cj)
    inv_cj=self.inv(x_cj)
    x_tps=self.MaskNet(gaussian_tps)
    inv_tps=self.inv(x_tps)
    inv_tps=inv_tps.repeat(1, 3, 1, 1)
    x_background=self.hadamard(inv_tps, img_tps)
    x_background=self.BG_Net(x_background)
    x_cj=x_cj.repeat(1,3,1,1)
    x_foreground=self.hadamard(x_cj, x_foreground)

    inv_cj=inv_cj.repeat(1, 3, 1, 1)
    x_background=self.hadamard(inv_cj, x_background)
    out=x_foreground+x_background
    print(out.shape)
    
    return out

class PipelineTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """
    
    """
    Components:
      Pose encoder
      Appearance encoder
      MaskNet
      FG_decoder
      BGNet(a simple UNet???)
      Vgg(not updating its weight)
    """

    def __init__(self, args): #recieve args from somewhere else...
      self.args = args
      if args.is_video:
        pass
      else:
        self.root_dir=args.data_dir
        self.csv_file=args.csv_file
      
      self.device=torch.device("cuda" if args.cuda else "cpu")

      #############Get model and initialize training related attributes
      self.model = PipelineNet(args).to(self.device) #get model
      self.vgg=Vgg19(requires_grad=False).to(self.device)
      self.total_loss=0
      self.mse_loss=torch.nn.MSELoss()
      self.optimizer = Adam(self.model.parameters(), args.lr)

      # if args.training:
      #   pass
      # else:
      #   pass

    def train(self, args, dataloader=None):
      
      if dataloader is None:
        raise RuntimeError("no input data")

      for epoch in range(args.epochs):
        self.model.train()
        self.total_loss=0
        count=0

        for batch_id, (img_cj, img_tps, gt) in enumerate(dataloader):
          self.optimizer.zero_grad()
          img_cj=img_cj.to(self.device)
          img_tps=img_tps.to(self.device)

          

          reconstructed_img=self.model(img_cj, img_tps)
          reconstructed_perception = self.vgg(reconstructed_img)
          groundtruth_perception=self.vgg(gt)
          
          self.total_loss += self.args.relu1_2_w * self.mse_loss(reconstructed_perception.relu1_2,\
                                                                 groundtruth_perception.relu1_2)
          self.total_loss += self.args.relu2_2_w * self.mse_loss(reconstructed_perception.relu2_2, \
                                                                 groundtruth_perception.relu2_2)
          self.total_loss += self.args.relu3_2_w * self.mse_loss(reconstructed_perception.relu3_2, \
                                                                 groundtruth_perception.relu3_2)
          self.total_loss += self.args.relu4_2_w * self.mse_loss(reconstructed_perception.relu4_2, \
                                                                 groundtruth_perception.relu4_2)
          self.total_loss.backward()
          
          self.optimizer.step()

          if (batch_id + 1) % args.log_interval == 0:
            mesg = "{}\tEpoch {}:\t[{}/{}]\t\ttotal: {:.6f}".format(
                    time.ctime(), epoch + 1, count, self.data_len, (self.total_loss) / (batch_id + 1))
            plt.imshow(reconstructed_img.detach().numpy())
            print(mesg)
          if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                self.model.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(epoch) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(self.model.state_dict(), ckpt_model_path)
                self.model.to(self.device).train()
      
    
    def save(self, epoch):
         # save model
      self.model.eval().cpu()
      save_model_filename = "epoch_" + str(self.args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
      self.args.content_weight) + "_" + str(self.args.style_weight) + ".model"
      save_model_path = os.path.join(args.save_model_dir, save_model_filename)
      torch.save(transformer.state_dict(), save_model_path)

    ##################################################################
    # Helper functions
    ##################################################################
    
    def check_paths(self,args):
        try:
            if not os.path.exists(args.save_model_dir):
                os.makedirs(args.save_model_dir)
                if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
                    os.makedirs(args.checkpoint_model_dir)
        except OSError as e:
            print(e)
            sys.exit(1)
        
    def logging(self, batch_id):
      if (batch_id + 1) % self.args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(message)