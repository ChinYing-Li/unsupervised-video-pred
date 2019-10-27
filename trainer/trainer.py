#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 02:20:45 2019

@author: liqinying
"""

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pix2pix_model import Pix2PixModel
import torch
import torch.nn as nn

class PipelineNet(nn.Module):
  def __init__():
    super(PipelineNet,self).__init__(args)
    self.cj_PoseEnc=Pose_Enc()
    self.tps_PoseEnc=Pose_Enc()
    self.cj_MaskNet=MaskNet()
    self.tps_MaskNet=MaskNet()
    self.App_Encoder=App_Enc()
    self.FGDecoder=FGNet()
    self.BG_Net=BGNet()
    
  def forward(self,img_cj, img_tps):

    x_cj=cj_PoseEnc(img_cj)
    x_tps=tps_PoseEnc(tps_PoseEnc)
    # TO DO: gaussian_cj function
    gaussian_cj=Gaussian_Fit(x_cj)
    gaussian_tps=Guassian_Fit(x_tps)
    
    x_conflux=App_Encoder(x_tps)
    x_conflux=FGDecoder(torch.cat(x_conflux, gaussian_cj), dim=1)
   
    x_cj=cj_MaskNet(gaussian_cj)
    x_tps=tps_MaskNet(gaussian_tps)
    
    #x=x_cj+
    return(x)

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
      cj_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        #transforms.ColorJitter(brightness=args.jitter_brightness, contrast=args.jitter_contrast,\
        #                       saturation=args.jitter_saturation, hue=args.jitter_hue)
        # Crop around annotation
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
        ])
    
      #when doing experiment on image datasets, have to use thin-plate-spline to perturb pose
      tps_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        # Crop around annotation
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
        ])
      
      self.device=args.device
      
      cj_dataset = datasets.ImageFolder(args.dataset, cj_transform)#???
      tps_dataset=datasets.ImageFolder(args.dataset, tps_transform)
      merged_dataset=TensorDataset(cj_dataset, tps_dataset)
      cj_loader = DataLoader(cj_dataset, batch_size=args.batch_size)
      tps_loader = DataLoader(tps_dataset, batch_size=args.batch_size)
        
      self.vgg=Vgg19(requies_grad=False).to_device(self.device)
      #one dataset
      
      self.args = args
      self.pipeline = PipelineNet(args) #get model
      self.total_loss=0
      self.mse_loss=torch.nn.MSELoss()
      self.device=torch.device("cuda" if args.cuda else "cpu")
      self.optimizer = Adam(self.pipeline.parameters(), args.lr)
        
  
      if args.training:
        pass
      else:
        pass

    def train(self, args):
      self.total_loss=0
      reconstructed_img=()
      reconstructed_ = vgg(reconstructed_img)
      gt = vgg(gt_img)
      
      self.total_loss += args.relu1_2_w * mse_loss(reconstructed.relu1_2, gt.relu1_2)
      self.total_loss += args.relu2_2_w * mse_loss(reconstructed.relu2_2, gt.relu2_2)
      self.total_loss += args.relu3_2_w * mse_loss(reconstructed.relu3_2, gt.relu3_2)
      self.total_loss += args.relu4_2_w * mse_loss(reconstructed.relu4_2, gt.relu4_2)
      
      
    
    def save(self, epoch):
         # save model
      self.model.eval().cpu()
      save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
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
        
    def logging(self):
      if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(message)
            
    def preprocess_input(self, args):
      """
      
      """
      pass