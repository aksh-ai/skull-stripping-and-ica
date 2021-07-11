# Author: Akshay Kumaar M (aksh-ai)

# Necessary imports
import os
from tqdm import tqdm
import torch as th
import numpy as np
import pandas as pd
import nibabel as nib
import torchio as tio
from scipy import stats
from shutil import copyfile
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as FT
from torch.utils.tensorboard import SummaryWriter 
from ipywidgets import interact, interactive, IntSlider, ToggleButtons

def prepare_data(csv_path: str = None, out_dir: str = 'data') -> None:
    '''
    Prepares the data into a torch Dataset friendly manner that can be easily loaded and used

    Args:
      * csv_path: Path to the CSV file containing the NFBS file paths
      * out_dir: Saves the images into sub-directories under the given path

    Returns:
      * None
    '''
    # proceed only if the CSV file exists
    if os.path.exists(csv_path):
        # create subdirs list
        out_dirs = [os.path.join(out_dir, sub_dir) for sub_dir in ['images', 'labels', 'targets']]
        
        # create subdir under out dir 
        for out_dir in out_dirs:
            os.makedirs(out_dir, exist_ok=True)

        # read the csv file from path
        df = pd.read_csv(csv_path)

        # iteratively copy images from each class to the respective folders under out dir
        for i in range(len(df)):
            # get skull MRI, skull-stripped brain MRI, and labels (masks) path
            skull = df['skull'][i].split('\\')[-1]
            brain = df['brain'][i].split('\\')[-1]
            mask = df['mask'][i].split('\\')[-1]

            # copy files to subdirs
            copyfile(df['skull'][i], os.path.join(out_dirs[0], skull))
            copyfile(df['mask'][i], os.path.join(out_dirs[1], mask))
            copyfile(df['brain'][i], os.path.join(out_dirs[2], brain))

    # if CSV is not present, raise an exception
    else:
        raise Exception("Invalid CSV path defined")

def plot_histogram(axis, tensor: th.Tensor, num_positions: int = 100, label: str = None, alpha: float = 0.05, color: str = None) -> None:
    '''
    Plots histogram intensity value of the tensor passed

    Args:
      * axis: Axis of the plot in which the value should be plotted
      * tensor: Tensor containing image data
      * num_positions: Number of bins/positions to use for calculating histogram intensity
      * label: label to be used while plotting
      * alpha: Alpha value for plotting
      * color: Color of the plot line
    
    Returns:
      * None
    '''
    # convert to numpy
    values = tensor.numpy().ravel()
    # get gaussian KDE kernel
    kernel = stats.gaussian_kde(values)
    # get histogram positions
    positions = np.linspace(values.min(), values.max(), num=num_positions)
    # calculate histograms using he filter
    histogram = kernel(positions)
    # plot the value on the axis using extra arguments
    kwargs = dict(linewidth=1, color='black' if color is None else color, alpha=alpha)
    # assign label
    if label is not None:
        kwargs['label'] = label
    # plot
    axis.plot(positions, histogram, **kwargs)

def get_histogram_plot(dataset, use_histogram_landmarks: bool = False, landmarks_path: str = 'NFBS_histogram_landmarks.npy') -> None:
    '''
    Plots histogram intesity values for images in a dataset

    Args:
      * dataset: A subjects dataset containing images
      * use_histogram_landmarks: Whether to use histogram intensity normalization while plotting
      * landmarks_path: Histogram mean and std values as a npy file to be laoded on the transform
    
    Returns:
      * None
    '''
    # create subplots
    fig, ax = plt.subplots(dpi=100)
    
    # set title
    title= 'Histograms of samples in the dataset'
    
    # load histograms if normalization has to be done
    if use_histogram_landmarks: 
        histogram_transform = tio.HistogramStandardization({'mri': np.load(landmarks_path)})
        title = 'Histogram Corrected samples of the dataset'
        
    # plot the histogram values without standardization
    for sample in tqdm(dataset):
        if use_histogram_landmarks: 
            tensor = histogram_transform(sample)    
        
        plot_histogram(ax, tensor.mri.data, color='blue')
                    
    # set plot configs
    ax.set_xlim(-100, 2000)
    ax.set_ylim(0, 0.004)
    ax.set_title(title)
    ax.set_xlabel('Intensity')
    ax.grid()

    # plot histogram
    plt.show()

def get_tensorboard(log_path: str = "skull_stripping_logs") -> SummaryWriter:
    '''
    Gives a tensorboard summary writer that cna used with the program and tensorboard

    Args:
      * log_path: Log dir to save the tensorboard logs in
    
    Returns:
      * None
    '''
    return SummaryWriter(log_path)

def save_checkpoint(args_dict, path: str = 'models/skull_stripping_ckpt.pth') -> None:
    '''
    Saves the checkpoint as a pth file in the specified path and with specified arguments

    Args:
      * args_dict: dictionary with keys and values like state dict, loss value, etc
      * path: Path to save the pth file in

    Returns:
      * None
    '''
    th.save(args_dict, path)

def dice_coefficient(y_true: th.Tensor, y_pred: th.Tensor, smooth: float = 1.0) -> th.Tensor:
    '''
    Calculates dice coefficients of two tensors

    Args:
      * y_true: targets tensor
      * y_pred: predicted tensor
      * smooth: smoothing value to be applied (0 - 1)
    
    Returns:
      * dice: Dice coefficient as a tensor
    '''
    # flatten the tensors
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    # calculate intersection
    intersection = (y_pred * y_true).sum()
    # calculate dice score
    dice = (2. * intersection + smooth)/(y_pred.sum() + y_true.sum() + smooth)  

    # return the dice coefficient
    return dice

def jaccard_similarity(y_true: th.Tensor, y_pred: th.Tensor, smooth: float = 1.0) -> th.Tensor:
    '''
    Calculates IoU (jaccard similarity) of two tensors

    Args:
      * y_true: targets tensor
      * y_pred: predicted tensor
      * smooth: smoothing value to be applied (0 - 1)
    
    Returns:
      * iou: IOU value as a tensor
    '''
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    intersection = (y_pred * y_true).sum()
    total = (y_pred + y_true).sum()
    union = total - intersection 
    
    iou = (intersection + smooth)/(union + smooth)

    return iou

def get_eval_metrics(y_true: th.Tensor, y_pred: th.Tensor) -> (th.Tensor, th.Tensor):
    '''
    Calculates dice coefficient and jaccard similarity scores

    Args:
      * y_true: targets tensor
      * y_pred: predicted tensor

    Returns:
      * dice, iou: tuple of tensors containing dice and iou score
    '''
    return dice_coefficient(y_true, y_pred), jaccard_similarity(y_true, y_pred)

def train_histograms(images_path: str = 'data/images', landmarks_path: str = 'NFBS_histogram_landmarks.npy') -> None:
    '''
    Train histograms for the given dataset directory

    Args:
      * images_path: Directory path to the images
      * landmarks_path: Name of the file to store the histogram mean and std values as npy file
    
    Returns:
      * None
    '''
    landmarks = tio.HistogramStandardization.train(
                    images_path,
                    output_path=landmarks_path,
                )
    np.set_printoptions(suppress=True, precision=3)

def get_train_transforms(histogram_landmarks='NFBS_histogram_landmarks.npy'):
    '''
    Get the train transforms object for on-the-fly preprocessing and augmentations for the training set

    Args:
      * histogram_landmarks: File of the histogram landmarks .npy file
    
    Returns:
      * transforms: transformation object
    '''
    return tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1),
        tio.RandomMotion(p=0.3),
        tio.HistogramStandardization({'mri': np.load(histogram_landmarks)}),
        tio.RandomBiasField(p=0.3),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        tio.RandomNoise(p=0.49),
        tio.RandomFlip(),
        tio.OneOf({
            tio.RandomAffine(): 0.6,
            tio.RandomElasticDeformation(): 0.4,
        })
    ])

def get_validation_transforms(histogram_landmarks='NFBS_histogram_landmarks.npy'):
    '''
    Get the validation transforms object for on-the-fly preprocessing and augmentations for the validation set

    Args:
      * histogram_landmarks: File of the histogram landmarks .npy file
    
    Returns:
      * transforms: transformation object
    '''
    return tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1),
        tio.HistogramStandardization({'mri': np.load(histogram_landmarks)}),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean)
    ])

class IntensityNormalization(object):
    @staticmethod
    def __call__(x: np.array, mask: np.array = None) -> np.array:
        '''
        Normalizes intensity using mean and std
        '''
        if mask is not None:
            mask_data = mask
        else:
            mask_data = x == x

        logical_mask = mask_data > 0.
        
        mean = x[logical_mask].mean()
        std = x[logical_mask].std()
        
        normalized = (x - mean) / std
        
        return normalized

class HistogramEqualize(object):
    def __init__(self, bins: int = 20) -> None:
        self.bins = bins

    def __call__(self, image: np.array, bins: int = None) -> np.array:
        '''
        Applies histogram standardization on the given image array
        '''
        bins = self.bins if bins == None else bins
        image_histogram, bins = np.histogram(image.flatten(), bins, density=True)
        cdf = image_histogram.cumsum()
        cdf = 255 * cdf / cdf[-1]

        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

        return image_equalized.reshape(image.shape)

def apply_binary_mask(img: np.array or th.Tensor, mask: np.array or th.Tensor) -> np.array:
    '''
    Computes/extracts the layer in the image from the given mask/label

    Args:
      * img: Image array
      * mask: Label or mask array
    
    Returns:
      * array of the segmented portion
    '''
    if type(img) == th.Tensor and type(mask) == th.Tensor:
        img, mask = img.item(), mask.item()

    background = np.zeros_like(img)
    foreground = mask * img
    background = (1 - mask) * background
    return foreground + background 

def plot_single_image(img: np.array or str, load: bool = False, axis: int = 3) -> None:
    '''
    Plot single 3D image for interactive visualization

    Args:
      * img: Image array or path
      * load: If path is given, set load as True to load the image using nibabel
      * axis: Axis of the image that should be visualized (0 - Axis 1, 1 - Axis 2, 2 - Axis 3, 3 - All axes)
    
    Returns:
      * None
    '''
    if load:
        img = nib.load(img).get_fdata()[:, :, :, np.newaxis]
    
    def explore_3dimage(depth):
        plt.figure(figsize=(10, 5))
        
        if axis+1 == 1:
            plt.imshow(img[depth, :, :, :], cmap='gray')
            plt.title("Coronal View")
            plt.axis('off')
        elif axis+1 == 2:
            plt.imshow(img[:, depth, :, :], cmap='gray')
            plt.title("Axial View")
            plt.axis('off')
        elif axis+1 == 3:
            plt.imshow(img[:, :, depth, :], cmap='gray')
            plt.title("Sagittal View")
            plt.axis('off')
        else:
            plt.subplot(1, 3, 1)
            plt.imshow(img[depth, :, :, :], cmap='gray')
            plt.title("Coronal View")
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(img[:, depth, :, :], cmap='gray')
            plt.title("Axial View")
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(img[:, :, depth, :], cmap='gray')
            plt.title("Sagittal View")
            plt.axis('off')

    interact(explore_3dimage, depth=(0, img.shape[axis-1] - 1 if axis!=0 else img.shape[axis] - 1))

    plt.show()

def plot_multiple_images(images: list, load: bool = False, labels: list = ["MRI - Skull Layers", "Skull Stripped Brain Layers", "Mask Layers"], axis: int = 1) -> None:
    '''
    Plot single 3D image for interactive visualization

    Args:
      * img: Image array or path
      * load: If path is given, set load as True to load the image using nibabel
      * labels: Labels that has to displayed for each image
      * axis: Axis of the image that should be visualized (1 - Axis 1, 2 - Axis 2, 3 - Axis 3)
    
    Returns:
      * None
    '''
    if load:
        for i in range(len (images)):
            images[i] = nib.load(images[i]).get_fdata()[:, :, :, np.newaxis]
    
    def explore_3dimage(depth):
        plt.figure(figsize=(10, 5))
        for i, img in enumerate(images):
            plt.subplot(1, len(images), i+1)
            if axis == 1:
                plt.imshow(img[depth, :, :, :], cmap='gray')
            elif axis == 2:
                plt.imshow(img[:, depth, :, :], cmap='gray')
            else:
                plt.imshow(img[:, :, depth, :], cmap='gray')
            plt.title(labels[i])
            plt.axis('off')

    interact(explore_3dimage, depth=(0, images[0].shape[axis-1] - 1))

    plt.show()