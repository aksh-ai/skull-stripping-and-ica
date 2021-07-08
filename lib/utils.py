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

plt.rcParams['axes.facecolor'] = 'black'

def prepare_data(csv_path: str = None, out_dir: str = 'data') -> None:
    if os.path.exists(csv_path):
        out_dirs = [os.path.join(out_dir, sub_dir) for sub_dir in ['images', 'labels', 'targets']]
        
        for out_dir in out_dirs:
            os.makedirs(out_dir, exist_ok=True)

        df = pd.read_csv(csv_path)

        for i in range(len(df)):
            skull = df['skull'][i].split('\\')[-1]
            brain = df['brain'][i].split('\\')[-1]
            mask = df['mask'][i].split('\\')[-1]

            copyfile(df['skull'][i], os.path.join(out_dirs[0], skull))
            copyfile(df['mask'][i], os.path.join(out_dirs[1], mask))
            copyfile(df['brain'][i], os.path.join(out_dirs[2], brain))

    else:
        raise Exception("Invalid CSV path defined")

def plot_histogram(axis, tensor, num_positions=100, label=None, alpha=0.05, color=None):
    values = tensor.numpy().ravel()
    kernel = stats.gaussian_kde(values)
    positions = np.linspace(values.min(), values.max(), num=num_positions)
    histogram = kernel(positions)
    kwargs = dict(linewidth=1, color='black' if color is None else color, alpha=alpha)
    if label is not None:
        kwargs['label'] = label
    axis.plot(positions, histogram, **kwargs)

def get_histogram_plot(dataset, use_histogram_landmarks=False, landmarks_path='NFBS_histogram_landmarks.npy'):
    fig, ax = plt.subplots(dpi=100)
    
    title= 'Histograms of samples in the dataset'
    
    if use_histogram_landmarks: 
        histogram_transform = tio.HistogramStandardization({'mri': np.load(landmarks_path)})
        title = 'Histogram Corrected samples of the dataset'
        
    for sample in tqdm(dataset):
        if use_histogram_landmarks: 
            tensor = histogram_transform(sample)    
        
        plot_histogram(ax, tensor.mri.data, color='blue')
                    
    ax.set_xlim(-100, 2000)
    ax.set_ylim(0, 0.004)
    ax.set_title(title)
    ax.set_xlabel('Intensity')
    ax.grid()

    plt.show()

def get_tensorboard(log_path="skull_stripping_logs"):
    return SummaryWriter(log_path)

def save_checkpoint(args_dict, path='models/skull_stripping_ckpt.pth'):
    th.save(args_dict, path)

def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    intersection = (y_pred * y_true).sum()                            
    dice = (2. * intersection + smooth)/(y_pred.sum() + y_true.sum() + smooth)  

    return dice

def jaccard_similarity(y_true, y_pred, smooth=1.0):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    intersection = (y_pred * y_true).sum()
    total = (y_pred + y_true).sum()
    union = total - intersection 
    
    iou = (intersection + smooth)/(union + smooth)

    return iou

def get_eval_metrics(y_true, y_pred):
    return dice_coefficient(y_true, y_pred), jaccard_similarity(y_true, y_pred)

def train_histograms(images_path: str = 'data/images', landmarks_path: str = 'NFBS_histogram_landmarks.npy'):
    landmarks = tio.HistogramStandardization.train(
                    images_path,
                    output_path=landmarks_path,
                )
    np.set_printoptions(suppress=True, precision=3)
    print('\nTrained landmarks:', landmarks)

def get_train_transforms(histogram_landmarks='NFBS_histogram_landmarks.npy'):
    return tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1),
        # tio.CropOrPad((128, 128, 128)),
        # tio.Resample(4),
        # tio.CropOrPad((64, 64, 64)),
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
    return tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1),
        # tio.CropOrPad((128, 128, 128)),
        # tio.Resample(4),
        # tio.CropOrPad((64, 64, 64)),
        tio.HistogramStandardization({'mri': np.load(histogram_landmarks)}),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean)
    ])

class IntensityNormalization(object):
    @staticmethod
    def __call__(x: np.array, mask: np.array = None) -> np.array:
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
        bins = self.bins if bins == None else bins
        image_histogram, bins = np.histogram(image.flatten(), bins, density=True)
        cdf = image_histogram.cumsum()
        cdf = 255 * cdf / cdf[-1]

        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

        return image_equalized.reshape(image.shape)

def dice_ratio(seg, gt):
    seg = seg.flatten()
    
    seg[seg > 0.5] = 1
    seg[seg <= 0.5] = 0

    gt = gt.flatten()
    gt[gt > 0.5] = 1
    gt[gt <= 0.5] = 0

    same = (seg * gt).sum()

    dice = 2 * float(same) / float(gt.sum() + seg.sum())

    return dice

def apply_binary_mask(img: np.array or th.Tensor, mask: np.array or th.Tensor) -> np.array:
    if type(img) == th.Tensor and type(mask) == th.Tensor:
        img, mask = img.item(), mask.item()

    background = np.zeros_like(img)
    foreground = mask * img
    background = (1 - mask) * background
    return foreground + background 

def plot_single_image(img: np.array or str, load: bool = False, axis: int = 3) -> None:
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

    interact(explore_3dimage, depth=(0, img.shape[axis-1] - 1))

    plt.show()

def plot_multiple_images(images: list, load: bool = False, labels: list = ["MRI - Skull Layers", "Skull Stripped Brain Layers", "Mask Layers"], axis: int = 1) -> None:
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