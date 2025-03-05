import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def training(model, device, train_loader, val_loader, 
             loss_func, optimizer, scheduler=None, epochs=10,
             model_path=".", model_name='model', plot_loss=True):
    
    # List of training losses for each epoch
    loss_train_epoch = []
    # List of validation losses for each epoch
    loss_val_epoch = []

    # Move model to device (gpu or cpu)
    model.to(device)

    for epoch in range(epochs):

        # Training loss for each batch
        loss_train_batch = 0
        # Index of the batch being processed
        batch_idx = 0

        # Model in training mode
        model = model.train()
        for X, Y in train_loader:
            # Move data and labels to device
            X = X.to(device)
            Y = Y.to(device)

            # Set gradients of optimizer to zero for each batch
            optimizer.zero_grad()

            # Forward pass (Make predictions for this batch)
            pred = model(X)

            # Compute the loss and its gradients
            loss = loss_func(pred, Y)
            loss.backward()

            # Update weights
            optimizer.step()

            # Gather data and report
            loss_train_batch += loss.item()
            batch_idx += 1           
            loss_on_batch = loss_train_batch / batch_idx
            sys.stdout.write('\r' + f"epoch = {epoch + 1}/{epochs}, batch = {batch_idx}, loss_train = {loss_on_batch:0.4f}")

        # Add training loss of each epoch to List
        loss_train_epoch.append(loss_on_batch)
        
        # scheduler for learning rate
        scheduler.step()

        # Model in evaluation mode
        model.eval()
        with torch.no_grad():
            # Validation loss for each batch
            loss_val_batch = 0
            # Index of the batch being processed
            val_batch_idx = 0

            for X, Y in val_loader:
                X = X.to(device)
                Y = Y.to(device)

                pred = model(X)
                loss = loss_func(pred, Y)

                loss_val_batch += loss.item()
                val_batch_idx += 1
                val_loss_on_batch = loss_val_batch / val_batch_idx
        # Add validation loss of each epoch to List
        loss_val_epoch.append(val_loss_on_batch)

        print('\r' + f"epoch = {epoch + 1}, Train loss = {loss_train_epoch[-1]:0.4f}, Val loss = {loss_val_epoch[-1]:0.4f}")

    # Save model
    PATH = os.path.join(model_path, f"{model_name}.pth")
    torch.save(model, PATH)

    if plot_loss:
        # Plot training and validation loss
        plt.figure()
        plt.plot(np.arange(epochs), loss_train_epoch, '-*', color='blue', label='Train')
        plt.plot(np.arange(epochs), loss_val_epoch, '-o', color='orange', label='Validation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.title(f'Final Validation Loss Value = {loss_val_epoch[-1]:.4f}')
