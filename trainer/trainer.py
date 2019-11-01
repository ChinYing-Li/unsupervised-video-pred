#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 02:20:45 2019

@author: liqinying
"""

from networks.subnetworks import Pose_Enc, App_Enc, MaskNet, BGNet, FG_Dec
from networks.gaussian_fitting import Gaussian_Fit
from networks.vgg import Vgg19

from utils.nn_utils import hadamard

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
###
import utils.preprocessing as img_dt # where class ImageLandmarksDataset is defined
###
from torchvision import transforms

import os
import time

class PipelineNet(nn.Module):
  def __init__(self, args):
    super(PipelineNet,self).__init__()
    
    part_activations = args.activation_maps
    apperance_dim = args.appearance_dim
    norm = 'Batch'

    # For SPADE.
    args.semantic_nc = args.activation_maps

    self.PoseEncoder=Pose_Enc(3, part_activations, norm=norm)
    self.MaskNet=MaskNet(part_activations, norm=norm)
    self.App_Encoder=App_Enc(part_activations, apperance_dim, norm=norm)
    self.FGDecoder=FG_Dec(args)
    self.BG_Net=BGNet(3)
    self.Gaussian_fit=Gaussian_Fit(part_activations)

  def forward(self,img_cj, img_tps):

    x_cj=self.PoseEncoder(img_cj)
    x_tps=self.PoseEncoder(img_tps)
    
    gaussian_cj=self.Gaussian_fit(x_cj)
    gaussian_tps=self.Guassian_fit(x_tps)
    
    x_conflux=self.App_Encoder(x_tps)
    x_conflux=self.FGDecoder(torch.cat(x_conflux, gaussian_cj), dim=1)
   
    x_cj=MaskNet(gaussian_cj)
    x_tps=MaskNet(gaussian_tps)

    print(x_cj, x_tps)
    
    return

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
        self.image_size=args.image_size
      
      self.device=torch.device("cuda" if args.cuda else "cpu")

      #############Get model and initialize training related attributes
      self.model = PipelineNet(args).to(self.device).double() #get model
      self.vgg=Vgg19(requires_grad=False).to(self.device)
      self.total_loss=0
      self.mse_loss=torch.nn.MSELoss()
      self.optimizer = Adam(self.model.parameters(), args.lr)

      #########Creating Dataloader###########
      cj_transform = transforms.Compose([
        img_dt.Rescale(self.image_size),
        img_dt.ColorJitter(70, 100, 120, .3),
        img_dt.ToTensor(),
        ])
    
     
      tps_transform = transforms.Compose([
        img_dt.Rescale(self.image_size),
        img_dt.ThinPlateSpline(4),
        img_dt.ToTensor(),
        ])

      gt_transform = transforms.Compose([
        img_dt.Rescale(self.image_size),
        img_dt.ToTensor(),
      ])
      
      cj_dataset = img_dt.ImageLandmarksDataset(self.root_dir, self.csv_file, cj_transform)#???
      tps_dataset=img_dt.ImageLandmarksDataset(self.root_dir, self.csv_file, tps_transform)
      gt_dataset=img_dt.ImageLandmarksDataset(self.root_dir, self.csv_file, gt_transform)

      data=TensorDataset(cj_dataset.get_tensor(), tps_dataset.get_tensor(), gt_dataset.get_tensor())
      self.loader=DataLoader(data, batch_size=args.batch_size)
      self.data_len=len(cj_dataset)
      # if args.training:
      #   pass
      # else:
      #   pass

    def train(self, args):

      for epoch in range(args.epochs):
        self.model.train()
        self.total_loss=0
        count=0

        for batch_id, (img_cj, img_tps, gt) in enumerate(self.loader):
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
            print(mesg)
          if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                self.model.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(self.model.state_dict(), ckpt_model_path)
                self.model.to(self.device).train()
      
    
    def save(self, epoch):
         # save model
      self.model.eval().cpu()
      save_model_filename = "epoch_" + str(self.args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
      args.content_weight) + "_" + str(args.style_weight) + ".model"
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