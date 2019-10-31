import torch
import os
import argparse
import time
from .trainer import PipelineTrainer
#####
#get arguments, import PipelineTrainer
#
#####

def main():
  #get arguments
  main_arg_parser = argparse.ArgumentParser(description="parser for Pipeline")
  subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

  train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
  train_arg_parser.add_argument("--is_video", type=bool, required=True, default=False,
                                  help="if the dataset is a video dataset, then set to True; if is an image dataset\
                                  set to False")
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
  trainer.train()

  # save model
  trainer.model.eval().cpu()
  save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
      args.content_weight) + "_" + str(args.style_weight) + ".model"
  save_model_path = os.path.join(args.save_model_dir, save_model_filename)
  torch.save(trainer.model.state_dict(), save_model_path)

  print("\nDone, trained model saved at", save_model_path)

if __name__=="__main__":
  main()