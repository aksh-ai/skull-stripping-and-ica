# Author: Akshay Kumaar M (aksh-ai)

# Necessary imports
import os
import time
import torchio as tio
import torch as th
from tqdm import tqdm
from lib.utils import *
import matplotlib.pyplot as plt

def train(train_loader, valid_loader, model, optimizer, criterion, epochs, device, scheduler=None, experiment=None, checkpoint=True, verbose=320, model_path='models/residual_unet'):
    '''
    Function to run the training operation on the model for the given dataset with evaluation

    Args:
      * train_loader: Data loader of the training set
      * valid_loader: Data loader of the validation/testing set
      * model: Model
      * optimizer: Optimizer
      * criterion: Loss function
      * epochs: Number of iterations to train the model
      * device: Device in which the training process should run on
      * scheduler: Scheduler for learning rate scheduling
      * experiment: Tensorboard summary writer object
      * checkpoint: Whether to checkpoint model file every for iteration
      * verbose: An integer value based on which the trianing stats will be printed
      * model_path: A path to save the model checkpoints in

    Returns:
      * train_loss, test_loss: Lists containing the training loss and validation loss values
    '''
    # empty lists to store training and validation losses in
    train_loss, test_loss = [], []
    
    # get starting tie of training
    overall_start = time.time()

    # get length of the dataset
    train_data_len = len(train_loader)
    valid_data_len = len(valid_loader)

    # start the training loop
    for epoch in range(1, epochs+1):
        # set mdoel to trianing mode
        model.train()

        print(f"Epoch [{epoch}/{epochs}]")

        # epoch starting time
        e_start = time.time()

        # empty lists to store losses in steps
        tt, tv = [], []

        # train in steps for each batch
        for b, batch in enumerate(train_loader):
            # set the data to specified device
            X_train = batch['mri'][tio.DATA].data.to(device)
            y_train = batch['brain'][tio.DATA].data.to(device)

            # zero optimizer gradients (zero accumulation, do this out of the step loop for gradient accumulation)
            optimizer.zero_grad()

            # forward apss the image
            y_pred = model(X_train)

            # calculate loss
            loss = criterion(y_pred, y_train)

            # append step loss
            tt.append(loss.item())

            # back propogate gradients
            loss.backward()

            # optimize model and scheduler a step
            optimizer.step()
            if scheduler != None: scheduler.step()

            # print stats
            if (b + 1) % verbose == 0 or (b + 1) == 1 or (b + 1) == train_data_len:
                dice, iou = get_eval_metrics(y_pred=y_pred, y_true=y_train)
                print(f"Train - Batch [{b+1:6d}/{train_data_len}] | Loss: {tt[-1]:.6f} | Dice Coefficient: {dice.item():.6f} | Jaccard (IoU) Score: {iou.item():.6f}")

                # log step loss in tensorboard
                if experiment:
                    experiment.add_scalar('training_loss_in_steps', tt[-1], epoch * train_data_len + b)

        # append average value of step loss as epoch loss and log it in tensorboard
        train_loss.append(th.mean(th.tensor(tt)))
        if experiment:
            experiment.add_scalar('training_loss_per_epoch', train_loss[-1], epoch * train_data_len + b)
        
        # model evaluation mode
        model.eval()

        # test in steps for each batch without calculating gradients
        with th.no_grad():
            for b, batch in enumerate(valid_loader):
                # set the data to specified device
                X_test= batch['mri'][tio.DATA].data.to(device)
                y_test = batch['brain'][tio.DATA].data.to(device)

                # forward pass the image
                y_pred = model(X_test)

                # calculate loss
                loss = criterion(y_pred, y_test)

                # append the step loss
                tv.append(loss.item())

                # print stats
                if (b + 1) % verbose == 0 or (b + 1) == 1 or (b + 1) == valid_data_len:
                    dice, iou = get_eval_metrics(y_pred=y_pred, y_true=y_test)
                    print(f"Validation - Batch [{b+1:6d}/{valid_data_len}] | Loss: {tv[-1]:.6f} | Dice Coefficient: {dice.item():.6f} | Jaccard (IoU) Score: {iou.item():.6f}")

                    # log step loss in tensorboard
                    if experiment:
                        experiment.add_scalar('validation_loss_in_steps', tv[-1], epoch * valid_data_len + b)
        
        # append average value of step loss as epoch loss and log it in tensorboard
        test_loss.append(th.mean(th.tensor(tv)))
        
        if experiment:
            experiment.add_scalar('validation_loss_per_epoch', test_loss[-1], epoch * valid_data_len + b)
        
        print(f"Epoch [{epoch}/{epochs}] - Duration {(time.time() - e_start)/60:.2f} minutes")

        # checkpoint the model
        if checkpoint:
            save_checkpoint({"epoch": epoch, "state_dict": model.state_dict(), "train_loss": train_loss[-1], "valid_loss": test_loss[-1]}, path=model_path + f"_{epoch}.pth")

    # get end time of overall training
    end_time = time.time() - overall_start    

    # print training summary
    print("\nTraining Duration {:.2f} minutes".format(end_time/60))
    print("GPU memory used : {} kb".format(th.cuda.memory_allocated()))
    print("GPU memory cached : {} kb".format(th.cuda.memory_reserved()))

    # return epoch train loss and test loss
    return train_loss, test_loss

def evaluate(test_loader, model, criterion, device):
    '''
    Evaluate the dataset with the given model and loss function

    Args:
      * test_loader: Dataloader for the evaluation set
      * model: Model
      * criterion: loss function
      * device: Device in which the evaluation should run on
    
    Returns:
      * loss, dice, iou - Average Loss value, dice score, and iou score as tensors
    '''
    # set model to evlauation mode
    model.eval()

    # empty lists to store the test loss, dice and iou scores
    test_loss, dice_score, iou_score = [], [], []

    # test in steps for each batch without calculating gradients
    with th.no_grad():
        for b, batch in enumerate(test_loader):
            # set the iamges and labels to specified device
            X_test= batch['mri'][tio.DATA].data.to(device)
            y_test = batch['brain'][tio.DATA].data.to(device)

            # forward pass
            y_pred = model(X_test)

            # calculate loss, dice, and iou score
            loss = criterion(y_pred, y_test)
            dice, iou = get_eval_metrics(y_pred=y_pred, y_true=y_test)

            # append the loss and scores to the corresponding list
            test_loss.append(loss.item())
            dice_score.append(dice.item())
            iou_score.append(iou.item())

    # return average loss, dice score, and iou score
    return th.mean(th.tensor(test_loss)), th.mean(th.tensor(dice_score)), th.mean(th.tensor(iou_score))

def infer(input_path, output_path, model, patch_size=64, overlap=16, batch_size=1, transforms=None, device="cuda", visualize=False, return_tensors=True):
    '''
    Run the end-to-end inference for a T1 Weighted MRI and store the skull-stripped MRI as a new file

    Args:
      * input_path: Input path to the MRI "nii" or "nii.gz" file
      * output_path: Ouptut path to store skull-stripped MRI file in "nii" or "nii.gz" format
      * model: Model to use for segmentation
      * patch_size: Patch size for sub-volume generation
      * overlap: Overlap size for aggregation of patches
      * batch_size: Batch size for inference
      * transforms: Transforms to be applied on the image (by default it'll be applied automatically if not specified)
      * device: Device in which inference should run on
      * visualize: Visualize the skull stripped result
      * return_tensors: Return the tensors of predicted mask, skull-stripped image, original image
    
    Returns:
      * original, skull_stripped, mask - Tensors of actual image, skull-stripped image, and binary mask will be returned if return_tensors is selected
    '''
    # get default transforms for preprocessing and augmentation if transforms is not passed
    transforms = get_validation_transforms() if transforms is None else transforms

    # read image from input path and store it as a tio subject
    subject = transforms(tio.Subject(mri=tio.ScalarImage(input_path)))

    # sampler for patch generation
    grid_sampler = tio.inference.GridSampler(subject, patch_size, overlap)
    # dataloader for loading the patches
    patch_loader = th.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
    # aggregator for stitching the patches to whole image
    aggregator = tio.inference.GridAggregator(grid_sampler)

    # set mdoel to evaluation mode
    model.eval()
    
    # don't calculate gradients during inference
    with th.no_grad():
      # run inference
      for batch in patch_loader:
          # put image to required device
          inputs = batch['mri'][tio.DATA].to(device)
          # get rid location for aggregation
          locations = batch[tio.LOCATION]
          # get mask prediction
          pred = model(inputs)
          # aggrgate the image to get the whole image
          aggregator.add_batch(pred, locations)
    
    # get the whole image
    foreground = aggregator.get_output_tensor()

    # extract the required skull-stripped image and denormalize using input image's mean and std
    mask_applied = apply_binary_mask(subject.mri.data.numpy(), foreground.data.numpy()) * subject.mri.data.numpy().std() + subject.mri.data.numpy().mean()

    # convert to SclarImage and save the skull-stripped image
    pred = tio.ScalarImage(tensor=th.tensor(mask_applied), affine=subject.mri.affine)
    pred.save(output_path)

    # visualize the skull-stripepd image
    if visualize:
      pred = tio.Subject(mri = pred)
      pred.plot()
      plt.show()

    # return as tensors if required
    if return_tensors:
        return subject.mri.data.numpy(), mask_applied, foreground.data.numpy()