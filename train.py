# Author: Akshay Kumaar M (aksh-ai)

# Necessary imports
import os
import torch
import random
import argparse
from torch import nn
import numpy as np
import torchio as tio
from lib.data import *
from lib.utils import *
from lib.losses import *
from lib.layers import *
from lib.models import ResidualUNET3D
from lib.runners import *
import matplotlib.pyplot as plt
from pathlib import Path 

def run_training(args):
    print("Residual UNET 3D - Skull Stripping & Brain Segmentation - Training & Evaluation")
    
    # get images and labels path from the args
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)

    # set the device to run the training on
    if args.device is None:
        device= "cuda" if torch.cuda.is_available() else "cpu"
        torch.backends.cudnn.benchmark = True 
    else:
        device = args.device

    # get the required loss function
    if args.loss in ['MSE', 'L1', 'smooth_L1', 'BCE', 'BCE_logits']:
        criterion = StandardSegmentationLoss(loss_type=args.loss, num_classes=args.output_channels)
    elif args.loss == 'dice':
        criterion = DiceLoss()
    elif args.loss == 'iou':
        criterion = IoULoss()
    else:
        raise Exception("Invalid loss function defined. Please choose one from: MSE, L1, smooth_L1, BCE, BCE_logits, dice, iou")

    # set batch size, patch size, test_size, and epochs
    batch_size = args.batch_size
    patch_size = args.patch_size
    test_size = args.test_size
    epochs = args.epochs

    # get model and log dir
    model_dir = args.model_dir
    log_dir = args.log_dir

    # make the dirs if they dont exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # train the histogram standardization on the dataset to get the mu and sigma values and save it as npy file
    if args.hist_landmarks is None:
        hist_landmarks = 'NFBS_histogram_landmarks.npy'
        train_histograms(images_path=[os.path.join(args.images_dir, file) for file in os.listdir(args.images_dir)], landmarks_path=hist_landmarks)
    else:
        hist_landmarks = args.hist_landmarks

    # read the images and labels path
    images = sorted(images_dir.glob('*.nii.gz'))
    labels = sorted(labels_dir.glob('*.nii.gz'))

    # set train transformations for preprocessing and augmentations
    train_transforms = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(1),
            tio.RandomMotion(p=0.3),
            tio.HistogramStandardization({'mri': np.load(hist_landmarks)}),
            tio.RandomBiasField(p=0.3),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            tio.RandomNoise(p=0.49),
            tio.RandomFlip(),
            tio.OneOf({
                tio.RandomAffine(): 0.6,
                tio.RandomElasticDeformation(): 0.4,
            })
        ])

    # set valid transformations for preprocessing and augmentations
    valid_transforms = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(1),
            tio.HistogramStandardization({'mri': np.load(hist_landmarks)}),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean)
        ])

    # get the trianing and validation set as random sub-volumes with desired arguments
    training_set, validation_set = load_datasets(images, labels, patch_size=patch_size, volume="patches", test_size=test_size, train_transforms=train_transforms, valid_transforms=valid_transforms, random_state=52)

    # set dataloaders for the datasets with required batch size
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size)

    print("Prepared dataset for training and validation.")

    # instantiate the model
    model = ResidualUNET3D(in_channels=args.input_channels, out_channels=args.output_channels)

    # set parralelization based on the device
    if "cuda" in device:
        model = torch.nn.DataParallel(model)
        strict = True
    else:
        strict = False

    print("\nLoaded Model.")

    # if load checkpoint
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device), strict=strict)
        print(f"Loaded model weights from {args.model_path}")

    # display model summary
    print(model) 

    total_parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total_parameters - trainable

    print(f"Number of parameters in the model: {total_parameters}")
    print(f"Number of trainable parameters in the model: {trainable}")
    print(f"Number of non-trainable parameters in the model: {non_trainable}")

    # get the required optimizer
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise Exception("Invalid Optimzier selected. Choose one from: Adam, RMSprop, SGD")

    # get scheduler if selected
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 4e-5)
    else:
        scheduler = None
    
    # get tensorboard if selected
    if args.tensorboard:
        tensorboard = get_tensorboard(args.log_dir)
    else:
        tensorboard = None

    print(f"\nTraining Parameters - Epochs: {epochs} | Optim: {args.optim} | Loss: {args.loss} | Device {device.upper()} | Sub-Volume (Patch) Sizes: {args.patch_size} | Batch Size: {args.batch_size}")
    print("\nStarted training...")

    # train the model
    train_loss, valid_loss = train(train_loader, valid_loader, model.to(device), optimizer, criterion.to(device), epochs, device, scheduler=scheduler, verbose=args.verbose, checkpoint=args.ckpt, model_path=os.path.join(model_dir, args.model_name))

    # Visualize the loss progress post training the model
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.title("Loss - Progress")
    plt.legend()
    plt.savefig(args.model_name + "_loss_progress.png")
    plt.show()

    # evaluate the model on validation set
    print("\nValidating...")
    loss, dice, iou = evaluate(valid_loader, model.to(device), criterion.to(device), device)
    print(f"Overall Validation Metrics - Loss: {loss:.6f} Dice: {dice:.6f} IOU: {iou:.6f}")

    # save the trained model
    torch.save(model.state_dict(), os.path.join(model_dir, args.model_name))

if __name__ == '__main__':
    # get argument parser
    parser = argparse.ArgumentParser()
    
    # set arguments
    parser.add_argument("-i", "--images_dir", required=True, type=str, help="Name of the directory containing the MRI files without skull-stripping")
    parser.add_argument("-l", "--labels_dir", required=True, type=str, help="Name of the directory containing the labels/masks of the MRI")
    parser.add_argument("-m", "--model_dir", required=False, type=str, default="models", help="Name of teh directory to store checkpoints and model files")
    parser.add_argument("-ic", "--input_channels", type=int, default=1, required=False, help="Input channels for the MRI images in the dataset")
    parser.add_argument("-oc", "--output_channels", type=int, default=1, required=False, help="Output channels for the segmented image")
    parser.add_argument("-d", "--device", type=str, required=False, default=None, help="Device on which the model should run for inference. Ex: cuda:0, cuda:1, cuda, cpu")
    parser.add_argument("-mn", "--model_name", type=str, required=False, default="residual_unet3d_MSE", help="Name of the model's weights dict. Ex: 'models/segmentation.pth'")
    parser.add_argument("-p", "--patch_size", type=int, required=False, default=64, help="Size for generation of sub-volumes/patches from image. Ex: 32, 64, 128, etc")
    parser.add_argument("-b", "--batch_size", type=int, required=False, default=4, help="Batch size for inference. Ex: 1, 2, 32, 64, etc")
    parser.add_argument("-hl", "--hist_landmarks", type=str, required=False, default=None, help="Name of the file to store histogram normalization metrics as a numpy array")
    parser.add_argument("-e", "--epochs", type=int, required=False, default=5, help="Number of times to run the training loop")
    parser.add_argument("-ls", "--loss", type=str, required=False, default='MSE', help="Loss function. Choose one from: MSE, L1, smooth_L1, BCE, BCE_logits, dice, iou")
    parser.add_argument("-opt", "--optim", type=str, required=False, default='Adam', help="Optimization methods. Choose one from: Adam, RMSprop, SGD")
    parser.add_argument("-lr", "--lr", type=float, required=False, default=1e-3, help="Learning rate")
    parser.add_argument("-s", "--scheduler", type=bool, required=False, default=True, help="To use lr scheduler for training")
    parser.add_argument("-t", "--test_size", type=float, required=False, default=0.2, help="Train Test split percentage as a float value")
    parser.add_argument("-tb", "--tensorboard", type=bool, required=False, default=True, help="Log losses for Tensorboard usage")
    parser.add_argument("-c", "--ckpt", type=bool, required=False, default=True, help="Checkpoint model weights for training")
    parser.add_argument("-log", "--log_dir", type=str, required=False, default="skull-stripping-train-logs", help="Directory to log for tensorboard")
    parser.add_argument("-v", "--verbose", type=int, required=False, default=400, help="Display training log using this number while training")
    parser.add_argument("-mp", "--model_path", required=False, default=None, type=str, help="Model weights will be loaded from this file and then trianed on the dataset")

    # call the trianing function with prased arguments
    run_training(parser.parse_args())

    # clean up
    exit()