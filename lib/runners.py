import os
import time
import torchio as tio
import torch as th
from tqdm import tqdm
from lib.utils import *
import matplotlib.pyplot as plt

def train(train_loader, valid_loader, model, optimizer, criterion, epochs, device, scheduler=None, experiment=None, checkpoint=True, verbose=320, model_path='models/residual_unet'):
    model.train()

    train_loss, test_loss = [], []
    
    overall_start = time.time()

    train_data_len = len(train_loader)
    valid_data_len = len(valid_loader)

    for epoch in range(1, epochs+1):
        print(f"Epoch [{epoch}/{epochs}]")

        e_start = time.time()

        tt, tv = [], []

        for b, batch in enumerate(train_loader):
            X_train = batch['mri'][tio.DATA].data.to(device)
            y_train = batch['brain'][tio.DATA].data.to(device)

            optimizer.zero_grad()

            y_pred = model(X_train)

            loss = criterion(y_pred, y_train)

            tt.append(loss.item())

            loss.backward()

            optimizer.step()
            if scheduler != None: scheduler.step()

            if (b + 1) % verbose == 0 or (b + 1) == 1 or (b + 1) == train_data_len:
                dice, iou = get_eval_metrics(y_pred=y_pred, y_true=y_train)
                print(f"Train - Batch [{b+1:6d}/{train_data_len}] | Loss: {tt[-1]:.6f} | Dice Coefficient: {dice.item():.6f} | Jaccard (IoU) Score: {iou.item():.6f}")

                if experiment:
                    experiment.add_scalar('training_loss_in_steps', tt[-1], epoch * train_data_len + b)

        train_loss.append(th.mean(th.tensor(tt)))
        if experiment:
            experiment.add_scalar('training_loss_per_epoch', train_loss[-1], epoch * train_data_len + b)
        
        model.eval()

        with th.no_grad():
            for b, batch in enumerate(valid_loader):
                X_test= batch['mri'][tio.DATA].data.to(device)
                y_test = batch['brain'][tio.DATA].data.to(device)

                y_pred = model(X_test)

                loss = criterion(y_pred, y_test)

                tv.append(loss.item())

                if (b + 1) % verbose == 0 or (b + 1) == 1 or (b + 1) == valid_data_len:
                    dice, iou = get_eval_metrics(y_pred=y_pred, y_true=y_test)
                    print(f"Validation - Batch [{b+1:6d}/{valid_data_len}] | Loss: {tv[-1]:.6f} | Dice Coefficient: {dice.item():.6f} | Jaccard (IoU) Score: {iou.item():.6f}")

                    if experiment:
                        experiment.add_scalar('validation_loss_in_steps', tv[-1], epoch * valid_data_len + b)
        
        test_loss.append(th.mean(th.tensor(tv)))
        
        if experiment:
            experiment.add_scalar('validation_loss_per_epoch', test_loss[-1], epoch * valid_data_len + b)
        
        print(f"Epoch [{epoch}/{epochs}] - Duration {(time.time() - e_start)/60:.2f} minutes")

        if checkpoint:
            save_checkpoint({"epoch": epoch, "state_dict": model.state_dict(), "train_loss": train_loss[-1], "valid_loss": test_loss[-1]}, path=model_path + f"_{epoch}.pth")

    end_time = time.time() - overall_start    

    # print training summary
    print("\nTraining Duration {:.2f} minutes".format(end_time/60))
    print("GPU memory used : {} kb".format(th.cuda.memory_allocated()))
    print("GPU memory cached : {} kb".format(th.cuda.memory_reserved()))

    return train_loss, test_loss

def evaluate(test_loader, model, criterion, device):
    model.eval()

    test_loss, dice_score, iou_score = [], [], []

    for b, batch in enumerate(test_loader):
        X_test= batch['mri'][tio.DATA].data.to(device)
        y_test = batch['brain'][tio.DATA].data.to(device)

        y_pred = model(X_test)

        loss = criterion(y_pred, y_test)
        dice, iou = get_eval_metrics(y_pred=y_pred, y_true=y_test)

        test_loss.append(loss.item())
        dice_score.append(dice.item())
        iou_score.append(iou.item())

    return th.mean(th.tensor(test_loss)), th.mean(th.tensor(dice_score)), th.mean(th.tensor(iou_score))

def infer(input_path, output_path, model, patch_size=64, overlap=16, batch_size=1, transforms=None, device="cuda", visualize=False, return_tensors=True):
    transforms = get_validation_transforms() if transforms is None else transforms

    subject = transforms(tio.Subject(mri=tio.ScalarImage(input_path)))

    # sampler
    grid_sampler = tio.inference.GridSampler(subject, patch_size, overlap)
    # dataloader
    patch_loader = th.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
    # aggregator
    aggregator = tio.inference.GridAggregator(grid_sampler)

    model.eval()
    
    with th.no_grad():
      for batch in patch_loader:
          inputs = batch['mri'][tio.DATA].to(device)
          locations = batch[tio.LOCATION]
          probabilities = model(inputs)
          aggregator.add_batch(probabilities, locations)
    
    foreground = aggregator.get_output_tensor()

    mask_applied = apply_binary_mask(subject.mri.data.numpy(), foreground.data.numpy()) * subject.mri.data.numpy().std() + subject.mri.data.numpy().mean()

    pred = tio.ScalarImage(tensor=th.tensor(mask_applied), affine=subject.mri.affine)
    pred.save(output_path)

    if visualize:
      pred = tio.Subject(mri = pred)
      pred.plot()
      plt.show()

    if return_tensors:
        return subject.mri.data.numpy(), mask_applied, foreground.data.numpy()