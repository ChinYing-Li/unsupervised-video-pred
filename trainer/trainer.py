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
    self.args=args
    self.n_heatmaps = args.n_heatmaps
    self.n_appearance = args.n_appearance
    self.norm = 'Instance'
    
    self.semantic_nc = args.n_heatmaps#???
    
    self.PoseEncoder=Pose_Enc(3, self.n_heatmaps, norm=self.norm)
    self.MaskNet=MaskNet(self.n_heatmaps, norm=self.norm)
    self.App_Encoder=App_Enc(args, norm=self.norm)
    self.FGDecoder=FG_Dec(args)
    self.BG_Net=BGNet()
    self.heatmap_fit=heatmap_fit(self.args)
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
    
    mask_cj=self.MaskNet(gaussian_cj)
    inv_cj=self.inv(x_cj)
    mask_tps=self.MaskNet(gaussian_tps)
    inv_tps=self.inv(x_tps)
    inv_tps=inv_tps.repeat(1, 3, 1, 1)
    x_background=self.hadamard(inv_tps, img_tps)
    x_background=self.BG_Net(x_background)
    x_cj=mask_cj.repeat(1,3,1,1)
    x_foreground=self.hadamard(x_cj, x_foreground)
    
    inv_cj=inv_cj.repeat(1, 3, 1, 1)
    x_background=self.hadamard(inv_cj, x_background)
    out=(x_foreground+x_background)
    #print(out.shape)
    
    return out, mask_cj, mask_tps

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
                                
                self.optimizer = Adam(self.model.parameters(), args.lr, weight_decay=args.opt_wgtdcy)
                self.loss=nn.MSELoss()
                                
                self.data_len=400
    def train(self, args, dataloader=None):
        torch.autograd.set_detect_anomaly(True)
        if dataloader is None:
            raise RuntimeError("no input data")
                                            
        for epoch in range(args.epochs):
            self.model.train()

            count=0
                                                    
            for batch_id, (img_cj, img_tps, gt) in enumerate(dataloader):
                total_loss=0.
                self.optimizer.zero_grad()
                img_cj=img_cj.to(self.device)
                img_tps=img_tps.to(self.device)

                                                            
                                                            
                reconstructed_img, mask_cj, mask_tps=self.model(img_cj, img_tps)
                print(torch.max(reconstructed_img[0]))
                print(torch.min(reconstructed_img[0]))
                reconstructed_perception = self.vgg(reconstructed_img)
                groundtruth_perception=self.vgg(gt.to(self.device))

                total_loss += self.args.relu1_2_w * self.loss(reconstructed_perception.relu1_2,\
                                                                                                              groundtruth_perception.relu1_2)
                total_loss += self.args.relu2_2_w * self.loss(reconstructed_perception.relu2_2, \
                                                                                                                  groundtruth_perception.relu2_2)
                total_loss += self.args.relu3_2_w * self.loss(reconstructed_perception.relu3_2, \
                                                                                                                                                                groundtruth_perception.relu3_2)
                total_loss += self.args.relu4_2_w * self.loss(reconstructed_perception.relu4_2, groundtruth_perception.relu4_2)
                                                                                                                                                                    total_loss.backward(retain_graph=True)
                                                                                                                                                                        
                self.optimizer.step()
                                                                                                                                                                    
                if (batch_id + 1) % args.log_interval == 0:
                                                                                                                                                                            mesg = "{}\tEpoch {}:\t[{}/{}]\t\ttotal: {:.6f}".format(
                                                                                                                                                                                                                                    time.ctime(), epoch + 1, count, self.data_len, (total_loss) / (batch_id + 1))
                                                                                                                                                                                
                f = plt.figure()
                f.add_subplot(2,3, 1)
                plt.imshow(gt[0].detach().cpu().numpy().transpose(1,2,0))
                f.add_subplot(2,3, 2)
                plt.imshow(img_cj[0].detach().cpu().numpy().transpose(1,2,0))
                f.add_subplot(2,3, 3)
                plt.imshow(img_tps[0].detach().cpu().numpy().transpose(1,2,0))
                f.add_subplot(2,3, 4)
                plt.imshow(reconstructed_img[0].detach().cpu().numpy().transpose(1,2,0))
                f.add_subplot(2,3, 5)
                plt.imshow(mask_cj[0].detach().cpu().numpy().transpose(1,2,0))
                f.add_subplot(2,3, 6)
                plt.imshow(mask_tps[0].detach().cpu().numpy().transpose(1,2,0))
                plt.show(block=True)
                print(mesg)                                                                                                                                                       
                                                                                                                                                                                    
                if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                    self.model.eval().cpu()
                    ckpt_model_filename = "ckpt_epoch_" + str(epoch) + "_batch_id_" + str(batch_id + 1) + ".pth"
                    ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': total_loss},
                        ckpt_model_path)
                    self.model.to(self.device).train()
                                                                                                                                                                                                
                                                                                                                                                                                                ###not in use
                def save(self, epoch):
                # save model
                self.model.eval().cpu()
                save_model_filename = "epoch_" + str(self.args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
                args.content_weight) + "_" + str(args.style_weight) + ".model"
                                                                                                                                                                                                                                                                                                                   save_model_path = os.path.join(args.save_model_dir, save_model_filename)
                                                                                                                                                                                                                                                                                                                       torch.save(transformer.state_dict(), save_model_path)
                                                                                                                                                                                                                                                                                                                           