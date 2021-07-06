import os
import time
import torch as th
from tqdm import tqdm
from lib.utils import *

def train(train_loader, valid_loader, model, optimizer, criterion, epochs, device, scheduler=None, experiment=None, checkpoint=True, verbose=1, model_path='models/residual_unet.pth'):
    model.train()

    train_loss, test_loss = [], []
    
    overall_start = time.time()

    train_data_len = len(train_loader)
    valid_data_len = len(valid_loader)

    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}")

        e_start = time.time()

        for b, batch in enumerate(train_loader):
            X_train, y_train = batch.mri.data.to(device), batch.brain.data.to(device)

            optimizer.zero_grad()

            y_pred = model(X_train)

            loss = criterion(y_pred, y_train)
            loss.backward()

            optimizer.step()
            if scheduler != None: scheduler.step()

            if train_data_len % b == 0 or b == train_data_len:
                print(f"Batch [{b}/{train_data_len}] | Dice Loss: {loss.item():.6f}")

                if experiment:
                    experiment.add_scalar('training_loss_in_steps', loss.item(), epoch * train_data_len + b)

        train_loss.append(loss.item())
        experiment.add_scalar('training_loss_per_epoch', train_loss[-1], epoch * train_data_len + b)
        
        model.eval()

        for b, batch in enumerate(valid_loader):
            X_test, y_test = batch.mri.data.to(device), batch.brain.data.to(device)

            y_pred = model(X_test)

            loss = criterion(y_pred, y_test)

            if train_data_len % b == 0 or b == train_data_len:
                print(f"Batch [{b}/{train_data_len}] | Dice Loss: {loss.item():.6f}")

                if experiment:
                    experiment.add_scalar('testing_loss_in_steps', loss.item(), epoch * valid_data_len + b)
                    
        print(f"Epoch {epoch} - Duration {(time.time() - e_start)/60}:.2f minutes")

    return train_loss, test_loss