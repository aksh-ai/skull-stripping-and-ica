# utility functions
import cv2
import numpy as np
import nibabel as nib
import torchvision.transforms as T
import torchvision.transforms.functional as FT
# schedulers / callbacks
# checkpoint savers
# preprocess functions as transforms

class IntensityNormalization(object):
    def __init__(self, normalization_type="z_score"):
        self.normalization_type = normalization_type
    
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
    def __call__(self, image):
        # return FT.equalize(x)
        image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
        cdf = image_histogram.cumsum() # cumulative distribution function
        cdf = 255 * cdf / cdf[-1] # normalize

        # use linear interpolation of cdf to find new pixel values
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

        return image_equalized.reshape(image.shape)
    
class BiasFieldCorrection(object):
    def __call__(self, x):
        return x

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)

def z_score(img, mask=None):
    img_data = img.get_fdata()
    if mask is not None and not isinstance(mask, str):
        mask_data = mask.get_fdata()
    else:
        mask_data = img_data == img_data
    logical_mask = mask_data > 0.  
    mean = img_data[logical_mask].mean()
    std = img_data[logical_mask].std()
    normalized = nib.Nifti1Image((img_data - mean) / std, img.affine, img.header)
    return normalized