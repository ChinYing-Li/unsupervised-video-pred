import torch.nn.functional as F
import torch.nn as nn
from .utils import nn_utils

from .SPADE.model.networks.base_network import BaseNetwork
from .SPADE.models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock

class Pose_Enc(nn.Module):
    
    """
    Pose Encoder, of which the architecture is Unet-like.
    
    Arguments
        in_channels: The channels of each input. If input is RGB colored image, then in_channels=3
        n_features: The number of feature maps as output; The parameter K in the paper
        norm: The mode of normalization layer. Should be either "Batch" or "Instance"
    """
    
    def __init__(self, in_channels, n_features, norm=None, bilinear=False):
        super(Pose_Enc, self).__init__()
        self.norm=norm

        self.inc = nn_utils.inconv(in_channels, 64, norm=norm)
        self.down1 = nn_utils.down(64, 128, norm=self.norm)
        self.down2 = nn_utils.down(128, 256, norm=self.norm)
        self.down3 = nn_utils.down(256, 512, norm=self.norm)
        self.down4 = nn_utils.down(512, 512,norm=self.norm)
        self.up1 = nn_utils.up(1024, 256, bilinear=bilinear, norm=None)
        self.up2 = nn_utils.up(512, 128, bilinear=bilinear, norm=None)
        self.up3 = nn_utils.up(256, 64,bilinear=bilinear, norm=None)
        self.up4 = nn_utils.up(128, 64,bilinear=bilinear, norm=None)
        self.outc = nn_utils.outconv(64, n_features)

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
        return x
    
class App_Enc(nn.Module):
  """
    Appearance Encoder, of which the architecture is Unet-like.
    
    Arguments
        n_features: The number of feature maps as output; The parameter K in the paper
        n_appearance: The number of feature maps as output; The parameter K in the paper
        norm: The mode of normalization layer. Should be either "Batch" or "Instance"
  """  
  
  class App_Enc(nn.Module):
  """
    Appearance Encoder, of which the architecture is Unet-like.
    
    Arguments
        n_features: The channels of heatmaps. If input is RGB colored image, then in_channels=3
        !!!should output a vector that can be feed into the SPADE generator (FGDecoder)
        n_appearance: The number of feature maps as output; The parameter K in the paper
        norm: The mode of normalization layer. Should be either "Batch" or "Instance"
  """  
  
  def __init__(self, n_heatmaps, n_appearance, norm=None):
    super(App_Enc, self).__init__()
    #the input raw image has three channels
    self.n_heatmaps=n_heatmaps
    self.n_appearance=n_appearance

    self.inc=inconv(3,8, norm=norm)
    self.down1=down(8, 16, norm=norm)
    self.mid=down(16, 16, norm=norm)
    self.up1=up(32, 8, norm=norm)
    self.outc=outconv(8, n_appearance)

  
  def forward(self, img, heatmaps):
    #here, x is the (raw ) image???
    #produce vectors to be feed into SPADE 
    x1=self.inc(img)
    x2=self.down1(x1)
    x3=self.mid(x2)
    x=self.up1(x3,x2)
    x=self.outc(x)
    out=torch.Tensor(self.n_heatmaps, self.n_appearance)

    for i in range(self.n_heatmaps):

      #softmax normalize the (not gaussian fitted) heatmaps
      p=torch.nn.Softmax(x[i])
      p=torch.unsqueeze(p, 0)
      p=p.repeat(self.n_appearance, 1, 1)
      p=torch.mul(img, p)
      out[i]=torch.sum(torch.sum(p, 2), 1)
    print("appearance encoder output vector\n")
    print(out.shape)
    return out


class MaskNet(nn.Module):
    def __init__(self, n_filters, norm=None):
        super(MaskNet, self).__init__()
    
        self.inc=nn_utils.inconv(n_filters, 32)
        self.down1=nn_utils.down(32, 32)
        self.down2=nn_utils.down(32, 32)
        self.down3=nn_utils.down(32, 32)
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
            
        return x

class BGNet(nn.Module):
    """
        Mask Networks, of which the architecure is Unet-like
        Arguments
        n_features: of the semantic
        n_filter: The number of filters used at each layer
        kernel_size: the size of kernel (filter).
        """
    
    def __init__(self, n_features, norm=None):
        super(MaskNet, self).__init__()
        self.inc=nn_utils.inconv(n_features, 32, norm=norm)
        self.down1=nn_utils.down(32,32, norm=norm)
        self.down2=nn_utils.down(32,32, norm=norm)
        self.down3=nn_utils.down(32,32, norm=norm)
        #have to think about how to concatanete this...
        self.up1=nn_utils.up(32, 32, norm=norm)
        self.up2=nn_utils.up(32, 32, norm=norm)
        pass
    
    def forward(self, x):
        pass


##The architecture of the foreground decoder shall follow the generator of SPADE
class FG_Dec(BaseNetwork):
    #def modify_commandline_options(parser, is_train):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        args.norm_G='spectralspadebatch3x3'
        self.sw, self.sh = self.modified_downsampling(args)
        
        if self.n_appearance:
            nf = self.n_appearance # number of generator filter... let's just keep this
        else:
            nf=16
        
        self.fc = nn.Conv2d(self.args.n_activation, 16 * nf, 3, padding=1)
        
        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, args)
        
        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, args)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, args)
        
        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, args)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, args)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, args)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, args)
        
        final_nc = nf
        
        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        
        self.up = nn.Upsample(scale_factor=2)

    def modified_downsampling(self, args):
        sw=args.crop_size[0]//args.downsamp_ratio
        sh=round(sw / args.aspect_ratio)
        return sw, sh
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
    
    def forward(self, input, z=None):
        seg = input
        
        
        # we downsample segmap and run convolution
        x = F.interpolate(seg, size=(self.sh, self.sw)) #downsample by a factor of 8; should be modified
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
        x = self.up(x)
        x = self.up_3(x, seg)
        
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        
        return x
