import torch as th
from torch.utils.data import DataLoader
import torchio as tio
from torchio import Subject, ScalarImage, DATA, LOCATION
from lib.utils import apply_binary_mask
from lib.models import ResidualUNET3D
from lib.runners import infer
import numpy as np

input_path = "T1Img/sub-05/T1w MRI.nii"
output_path = "sub-05 - T1w MRI - skull stripped.nii.gz"

th.cuda.empty_cache()

device= "cuda" if th.cuda.is_available() else "cpu"

model = ResidualUNET3D(in_channels=1, out_channels=1)

if "cuda" in device:
    model = th.nn.DataParallel(model).to(device)
    strict = True
else:
    strict = False

model.load_state_dict(th.load("models/residual_unet3d_MSE.pth", map_location=device), strict=strict)

th.cuda.empty_cache()

transforms = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1),
        tio.HistogramStandardization({'mri': np.load('NFBS_histogram_landmarks.npy')}),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean)
    ])

infer(input_path, output_path, model, mode="patches", patch_size=64, overlap=16, batch_size=1, device=device, transforms=transforms, return_tensors=False)