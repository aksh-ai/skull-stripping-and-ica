import torch as th
from torch.utils.data import DataLoader
import torchio as tio
from torchio import Subject, ScalarImage, DATA, LOCATION
from lib.utils import get_validation_transforms, apply_binary_mask
from lib.models import ResidualUNET3D

input_path = "T1Img/sub-05/T1w MRI.nii"
output_path = "sub-05 - T1w MRI - skull stripped.nii.gz"

th.cuda.empty_cache()

device= "cuda" if th.cuda.is_available() else "cpu"

def infer(input_path, output_path, model, mode="patches", patch_size=64, overlap=16, batch_size=1, transforms=None, device="cuda", return_tensors=True):
    transforms = get_validation_transforms() if transforms is None else transforms

    subject = transforms(Subject(mri=ScalarImage(input_path)))

    # sampler
    grid_sampler = tio.infernece.GridSampler(subject, patch_size, overlap)
    # dataloader
    patch_loader = DataLoader(grid_sampler, batch_size=batch_size)
    # aggregator
    aggregator = tio.inference.GridAggregator(grid_sampler)

    model.eval()
    
    for batch in patch_loader:
        inputs = batch['mri'][DATA].to(device)
        locations = batch[LOCATION].to(device)
        probabilities = model(inputs)
        aggregator.add_batch(probabilities, locations)
    
    foreground = aggregator.get_output_tensor()
    # prediction = ScalarImage(tensor=foreground, affine=subject.mri.affine)

    mask_applied = apply_binary_mask(subject.mri.data.numpy(), foreground.data.numpy())

    pred = ScalarImage(tensor=th.tensor(mask_applied), affine=subject.mri.affine)
    pred.save(output_path)

    if return_tensors:
        return pred.data

model = ResidualUNET3D(in_channels=1, out_channels=1).to(device)
model.load_state_dict(th.load("models/residual_unet3d_MSE.pth"))

th.cuda.empty_cache()

infer(input_path, output_path, model, mode="patches", patch_size=64, overlap=16, batch_size=1, device="cuda", return_tensors=False)