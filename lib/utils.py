# utility functions
import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as FT
from ipywidgets import interact, interactive, IntSlider, ToggleButtons

# schedulers / callbacks
# checkpoint savers
# preprocess functions as transforms

class IntensityNormalization(object):
    @staticmethod
    def __call__(x, mask=None):
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
    @staticmethod
    def __call__(image, bins=20):
        image_histogram, bins = np.histogram(image.flatten(), bins, density=True)
        cdf = image_histogram.cumsum()
        cdf = 255 * cdf / cdf[-1]

        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

        return image_equalized.reshape(image.shape)

def apply_binary_mask(img, mask):
    res = img * mask
    res = np.where(mask >= 0.5, img, mask)
    return res

def plot_single_image(img):
    img = nib.load(img)
    
    def explore_3dimage(depth):
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 1, 1)
        plt.imshow(np.expand_dims(img.get_fdata(), axis=3)[:, :, depth, :], cmap='gray')
        plt.title("MRI")
        plt.axis('off')

    interact(explore_3dimage, depth=(0, img.shape[2] - 1));