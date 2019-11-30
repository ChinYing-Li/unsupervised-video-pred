import torch.nn.functional as F
import torch.nn as nn
from .utils import nn_utils
import torch
from .SPADE.model.networks.base_network import BaseNetwork
from .SPADE.models.networks.FGResnet import FGResnetBlock as FGResnetBlock

class Pose_Enc(nn.Module):
    
    """
    Pose Encoder, of which the architecture is Unet-like.
    
    Arguments
        in_channels: The channels of each input. If input is RGB colored image, then in_channels=3
        n_features: The number of feature maps as output; The parameter K in the paper
        norm: The mode of normalization layer. Should be either "Batch" or "Instance"
    """
    
    def __init__(self, in_channels, n_heatmaps, norm=None, bilinear=False):
        super(Pose_Enc, self).__init__()
        self.norm=norm

        #only removed nn_utils here
        self.inc = nn_utils.inconv(in_channels, 64, norm=norm)
        self.down1 = nn_utils.strdown(64, 128, norm=self.norm)
        self.down2 = nn_utils.strdown(128, 256, norm=self.norm)
        self.down3 = nn_utils.strdown(256, 512, norm=self.norm)
        self.down4 = nn_utils.strdown(512, 512,norm=self.norm)
        self.up1 = nn_utils.up(1024, 256, bilinear=bilinear, norm=None)
        self.up2 = nn_utils.up(512, 128, bilinear=bilinear, norm=None)
        self.up3 = nn_utils.up(256, 64,bilinear=bilinear, norm=None)
        self.up4 = nn_utils.up(128, 64,bilinear=bilinear, norm=None)
        self.outc = nn_utils.outconv(64, n_heatmaps)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        print("Pose encoder output\n")
        print(x.shape)

        #x is the ultimate ouput of Pose_Enc; 
        # x1 will be fed to Appearance encoder (if following Lorentz. et al.)
        return x
    
class App_Enc(nn.Module):
  """
    Appearance Encoder, of which the architecture is Unet-like.
    
    Arguments
        n_features: The channels of heatmaps. If input is RGB colored image, then in_channels=3
        !!!should output a vector that can be feed into the SPADE generator (FGDecoder)
        n_appearance: The number of feature maps as output; The parameter K in the paper
        norm: The mode of normalization layer. Should be either "Batch" or "Instance"
  """  
  
  def __init__(self, args, norm=None):
    super(App_Enc, self).__init__()
    #the input raw image has three channels
    self.n_heatmaps=args.n_heatmaps
    self.n_appearance=args.n_appearance
    self.batch_size=args.batch_size
    self.crop_size=args.crop_size[0]
    assert type(self.crop_size) is int, "something wrong with crop size"
    self.inc=nn_utils.inconv(3,8, norm=norm)
    self.down1=nn_utils.strdown(8, 16, norm=norm)
    self.mid=nn_utils.strdown(16, 16, norm=norm)
    self.up1=nn_utils.up(32, 8, norm=norm)
    self.up2=nn_utils.up(16, 4, norm=norm)
    self.outc=nn_utils.outconv(4, self.n_appearance)

  
  def forward(self, tps_img, map1=None, map2=None):
    if map1 is None or map2 is None:
      raise RuntimeError("no heatmaps fed to appearance encoder")
    img=tps_img
    raw_tps=map1
    fitted_cj=map2

    #this is the approach in Lorentz et al.
    #inp=torch.cat(img_enc, raw_tps, dim=1)

    x1=self.inc(img)
    x2=self.down1(x1)
    x3=self.mid(x2)
    x=self.up1(x3,x2)
    x=self.up2(x,x1)
    x=self.outc(x)
    print(x.shape)
    app_vec=torch.Tensor(self.batch_size, self.n_heatmaps, self.n_appearance).cuda()
    
    for b in range(self.batch_size):
      for i in range(self.n_heatmaps):
        #softmax normalize the (not gaussian fitted) heatmaps
        p=torch.nn.functional.softmax(raw_tps[b][i])
        p=torch.unsqueeze(p, 0)
        p=p.repeat(self.n_appearance, 1, 1)
        p=torch.mul(x[b], p)
        app_vec[b][i]=torch.sum(torch.sum(p, 2), 1)
    print("appearance encoder output vector\n")
    print(app_vec.shape)

    #appearance vec -- k c-dimensional vector
    #project to the fitted color jittered image
    app_vec=torch.unsqueeze(app_vec, -2)
    app_vec=torch.unsqueeze(app_vec, -2)

    app=app_vec.repeat(1, 1, self.crop_size, self.crop_size,  1)
    p=torch.unsqueeze(fitted_cj, -1)
    p=p.repeat(1, 1, 1, 1, self.n_appearance)
    x=torch.mul(p, app)
    x=torch.sum(x, dim=1)

    denominator=1.+torch.sum(fitted_cj, dim=1)
    denominator=denominator.unsqueeze(-1)

    denominator=denominator.repeat(1,1,1,self.n_appearance)
    #app_enc shape: batch_size x H x W x n_appearance
    # should we transpose this... i think we should
    app_enco=x/denominator
    app_enco=torch.transpose(app_enco, 2,3)
    app_enco=torch.transpose(app_enco,1,2)
    #app_enc shape: batch_size x n_apprance x H x W 
    print("app encoder output")
    print(app_enco.shape)
    return app_enco

class MaskNet(nn.Module):
    """
    Mask Networks, of which the architecure is Unet-like
    Arguments
        n_features: of the semantic
        n_filter: The number of filters used at each layer
        kernel_size: the size of kernel (filter).
    """
    
    def __init__(self, n_heatmaps, norm=None):
        super(MaskNet, self).__init__()
    
        self.inc=nn_utils.inconv(n_heatmaps, 32)
        self.down1=nn_utils.strdown(32, 32)
        self.down2=nn_utils.strdown(32, 32)
        self.down3=nn_utils.strdown(32, 32)
        self.up1=nn_utils.up_MaskNet(64, 32, bilinear=False)
        self.up2=nn_utils.up_MaskNet(64,32, bilinear=False)
        self.up3=nn_utils.up_MaskNet(64,32, bilinear=False)
        self.outc=nn_utils.outconv(32, 1)
    
    def forward(self, x):
        x1=self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)   
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        x=torch.sigmoid(x)

        print("masknet output\n")
        print(x.shape)
        return x

class BGNet(nn.Module):
  def __init__(self, bilinear=None, norm=None):
    super(BGNet, self).__init__()
    #input is 3-channel 
    
    self.incv=nn_utils.inconv(3, 32, norm=norm)
    self.down1 = nn_utils.strdown(32, 64, norm=norm)
    self.down2 = nn_utils.strdown(64, 128, norm=norm)
    self.down3 = nn_utils.strdown(128, 256, norm=norm)
    self.down4 = nn_utils.strdown(256, 512,norm=norm)
    self.up1 = nn_utils.up(512, 128, bilinear=bilinear, norm=norm)
    self.up2 = nn_utils.up(256, 64,bilinear=bilinear, norm=norm)
    self.up3 = nn_utils.up(128, 32, bilinear=bilinear, norm=norm)
    self.up4 = nn_utils.up(64, 16, bilinear=bilinear, norm=norm)
    self.outc = nn_utils.outconv(16, 3)

  def forward(self, inp):
    assert inp.shape[1]==3,"BGNet input should be 3-ch"
    x1=self.incv(inp)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)
    x = self.up1(x5, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up3(x, x1)
    x = self.outc(x)
    print("BGNet output vector\n")
    print(x.shape)
    return x


##The architecture of the foreground decoder shall follow the generator of SPADE
class FG_Dec(BaseNetwork):
    #def modify_commandline_options(parser, is_train):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        args.norm_G='spectralspadebatch3x3'
        self.sw, self.sh = self.modified_downsampling(args)

        if args.ngf:
            nf = args.ngf # number of generator filter... let's just keep this
        else:
            nf=16

        if args.use_vae:
            # In case of VAE, we will sample from random z vector
            
            #print(16 * nf * self.sw * self.sh)
            self.fc = nn.Linear(self.args.n_appearance, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            #we will be inputing heatmaps
            self.fc = nn.Conv2d(self.args.n_appearance, 16 * nf, 3, padding=1)
        
        self.head_0 = FGResnetBlock(16 * nf, 16 * nf, args)
        
        self.G_middle_0 = FGResnetBlock(16 * nf, 16 * nf, args)
        self.G_middle_1 = FGResnetBlock(16 * nf, 16 * nf, args)
        
        self.up_0 = FGResnetBlock(16 * nf, 8 * nf, args)
        self.up_1 = FGResnetBlock(8 * nf, 4 * nf, args)
        self.up_2 = FGResnetBlock(4 * nf, 3, args)
        #self.up_3 = FGResnetBlock(2 * nf, 1 * nf, args)
        
        final_nc = 3
        
        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        
        self.up = nn.Upsample(scale_factor=2)

    def modified_downsampling(self, args):
      sw=args.crop_size[0]//args.downsamp_ratio
      sh=round(sw / args.aspect_ratio)
      
      return round(sw), sh

    def compute_latent_vector_size(self, args):
        if args.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif args.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif args.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             args.num_upsampling_layers)

        sw = args.crop_size[0] // (2**num_up_layers)
        sh = round(sw / args.aspect_ratio)

        return sw, sh

    def forward(self, heatmaps, appearance_enco=None):
        #input is the heat map
        # z is the vectors from Appearence encoder 
        seg=heatmaps
        print("app enco")
        print(appearance_enco.shape)

        if self.args.use_vae:
            # we sample z from unit normal and reshape the tensor
            if appearance_enco is None:
                appearance_enco = torch.randn(input.size(0), self.args.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(appearance_enco)
            x = x.view(-1, 16 * self.args.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(appearance_enco, size=(self.sh, self.sw))
            x = self.fc(x)
        
        x = self.head_0(x, seg)
        
        x = self.up(x) #use this as well??
        x = self.G_middle_0(x, seg)
        
        x = self.G_middle_1(x, seg)
        
        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        #x = self.up(x)
        #x = self.up_3(x, seg)
        
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        print("App decoder output vector\n")
        print(x.shape)
        return x