import torch
import os
import argparse
import time
from trainer.trainer import PipelineTrainer

#####
#get arguments, import PipelineTrainer
#
#####

def main():
  #get arguments
  main_arg_parser = argparse.ArgumentParser(description="parser for Pipeline")
  subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

  train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
  train_arg_parser.add_argument("--is_video", default=False, action='store_false',
                                  help="if the dataset is a video dataset, then set to True; if is an image dataset\
                                  omit this argument.")
  train_arg_parser.add_argument("--data_dir", type=str, required=True,
                                  help="path to the directory that contrains the training dataset, the path should point to a folder\
                                       containing all the training images and a (optional) .csv file which contains\
                                       landmarks info")
  train_arg_parser.add_argument("--csv-file", type=str, default=None)
  train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
  train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
  train_arg_parser.add_argument("--batch_size", type=int, default=4,
                                  help="batch size for training, default is 4")
    
  ##do we need image-size??
  train_arg_parser.add_argument("--image-size", type=int, default=128,
                                  help="size of training images, default is 128 X 128")
  train_arg_parser.add_argument("--lr", type=float, default=1e-5,
                                  help="learning rate, default is 1e-5")
  train_arg_parser.add_argument("--specnorm", type=bool, default=True,
                                  help="Whether to apply spectral normalization to FGdecoder, default is True")
  train_arg_parser.add_argument("--input-channel", type=int, default=3,
                                  help="The channel in input images, default is 3")  
  #saving the result
  train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
  train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    
    #arguments for vgg
  train_arg_parser.add_argument("--relu1-2-w", type=float, default=1e5,
                                  help="weight for content-loss, default is 0.5")
  train_arg_parser.add_argument("--relu2-2-w", type=float, default=1e5,
                                  help="weight for content-loss, default is 0.5")
  train_arg_parser.add_argument("--relu3-2-w", type=float, default=1e5,
                                  help="weight for content-loss, default is 0.5")
  train_arg_parser.add_argument("--relu4-2-w", type=float, default=1e5,
                                  help="weight for content-loss, default is 0.5")

  # Training parameters
  train_arg_parser.add_argument("--activation_maps", type=int, default=10,
                                help="number of part activations maps for pose encoder.")
  train_arg_parser.add_argument("--appearance_dim", type=int, default=10,
                                help="number of appearance encoder output filters. Determines size of appearance vector.")
  # SPADE parameters
  train_arg_parser.add_argument('--norm_G', type=str, default='spectralspadebatch3x3', help='instance normalization or batch normalization')
  train_arg_parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
  train_arg_parser.add_argument('--norm_E', type=str, default='spectralinstance', help='instance normalization or batch normalization')

  # SPADE input/output sizes
  train_arg_parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
  train_arg_parser.add_argument('--preprocess_mode', type=str, default='scale_width_and_crop', help='scaling and cropping of images at load time.', choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
  train_arg_parser.add_argument('--load_size', type=int, default=1024, help='Scale images to this size. The final image will be cropped to --crop_size.')
  train_arg_parser.add_argument('--crop_size', type=int, default=512, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
  train_arg_parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
  train_arg_parser.add_argument('--label_nc', type=int, default=182, help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
  train_arg_parser.add_argument('--contain_dontcare_label', action='store_true', help='if the label map contains dontcare label (dontcare=255)')
  train_arg_parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

  # SPADE params for generator
  train_arg_parser.add_argument('--netG', type=str, default='spade', help='selects model to use for netG (pix2pixhd | spade)')
  train_arg_parser.add_argument('--ngf', type=int, default=16, help='# of gen filters in first conv layer') #(TODO): This was changed to 16, change back?
  train_arg_parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
  train_arg_parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
  train_arg_parser.add_argument('--z_dim', type=int, default=256,
                      help="dimension of the latent z vector")

  # SPADE params for instance-wise features
  train_arg_parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')
  train_arg_parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
  train_arg_parser.add_argument('--use_vae', action='store_true', help='enable training with an image encoder.')
    
    #logginf and checkpointing
  train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
  train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")

  eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation")
  eval_arg_parser.add_argument("--test-image", type=str, required=True,
                                 help="path to content image you want to reconstruct")
  eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
  eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
  eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")

  args = main_arg_parser.parse_args()

  #get trainer
  trainer=PipelineTrainer(args)
  trainer.train(args)

  # save model
  trainer.model.eval().cpu()
  save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
      args.content_weight) + "_" + str(args.style_weight) + ".model"
  save_model_path = os.path.join(args.save_model_dir, save_model_filename)
  torch.save(trainer.model.state_dict(), save_model_path)

  print("\nDone, trained model saved at", save_model_path)

if __name__=="__main__":
  main()