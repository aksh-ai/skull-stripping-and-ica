# Author: Akshay Kumaar M (aksh-ai)

# Necessary imports
import argparse
import torch as th
import torchio as tio
from lib.runners import infer
from lib.models import ResidualUNET3D

if __name__ == '__main__':
    # get argument parser
    parser = argparse.ArgumentParser()

    # set arguments
    parser.add_argument("-i", "--input", type=str, required=True, help="Input path for loading the skull-stripped brain image file. Ex: b1.nii.gz")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output path for saving the skull-stripped brain image file. Ex: outputs/b1.nii.gz")
    parser.add_argument("-d", "--device", type=str, required=False, default=None, help="Device on which the model should run for inference. Ex: cuda:0, cuda:1, cuda, cpu")
    parser.add_argument("-m", "--model_path", type=str, required=False, default="models/residual_unet3d_MSE_3.pth", help="Path to the model's weights dict. Ex: 'models/segmentation.pth'")
    parser.add_argument("-p", "--patch_size", type=int, required=False, default=64, help="Size for generation of sub-volumes/patches from image. Ex: 32, 64, 128, etc")
    parser.add_argument("-l", "--overlap_size", type=int, required=False, default=16, help="Sub-volume image aggregation overlap size for stitching patch of images to a whole image. Ex: 4, 18, 16 (Cannot be greater than patch size)")
    parser.add_argument("-b", "--batch_size", type=int, required=False, default=1, help="Batch size for inference. Ex: 1, 2, 32, 64, etc")
    parser.add_argument("-v", "--visualize", type=str, required=False, default=True, help="Visualize skull-stripped image of the brain: True or False")

    # parse the arguments
    args = parser.parse_args()

    # set the device to run the inference on
    if args.device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"
    else:
        device = args.device

    # instantiate the model
    model = ResidualUNET3D(in_channels=1, out_channels=1, optional_skip=True)

    # set parallelization if device is cuda
    if "cuda" in device:
        model = th.nn.DataParallel(model)
        # trained with dataparallel so set strict as true
        strict = True
    else:
        # set strict as false o load model weights without dataparallel module
        strict = False

    # load the model weights
    model.load_state_dict(th.load(args.model_path, map_location=device), strict=strict)
    
    print(f"Loaded Residual UNET 3D Model on {device.upper()}")
    print(f"Inference Parameters - Image: {args.input} | Sub-Volume (Patch) Sizes: {args.patch_size} | Overlap Size: {args.overlap_size} | Batch Size: {args.batch_size} | Visualization: {args.visualize}")

    # run the inference
    infer(args.input, args.output, model.to(device), patch_size=args.patch_size, overlap=args.overlap_size, batch_size=args.batch_size, device=device, visualize=args.visualize, return_tensors=False)
    
    print(f"Skull-stripped Image saved at: {args.output}")

    # clean up
    exit()