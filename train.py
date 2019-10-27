import argparser
def main():
    pass

def get_arguments():
  """
  what are the user input arguments we want?
  epochs=
  batch_size=16
  training=True 
  input_data=path/to/data
  output_data=path/to/store/the/result
  learning_rate=1e-5
  save_model_dir
  cuda=False
  n_activation=5 (the number of landmarks found; a.k.a K in the paper)
  spectral_in_FGdecoder=True (whether to apply spectral normalization in the foregound decoder)
  #input images are assumed to be 3-channeled
  """
    main_arg_parser = argparse.ArgumentParser(description="parser for Pipeline")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    # arguments 
    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder\
                                       containing another folder with all the training images")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--lr", type=float, default=1e-5,
                                  help="learning rate, default is 1e-5")
    train_arg_parser.add_argument("--specnorm", type=bool, default=True,
                                  help="Whether to apply spectral normalization to FGdecoder, default is True")
    
    #saving the result
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    
    #arguments for vgg
    train_arg_parser.add_argument("--relu1_2_w", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--relu2_2_w", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--relu3_2_w", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--relu4_2_w", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    
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

    return args