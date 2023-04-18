import os
import pandas as pd
import torch
from config import *
import time 
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from copy import deepcopy as dc

from evaluation import evaluation_metrics

    
def train_model(model, train_loader, val_loader, criterion, optimizer, name='Model'):
    # initialize a dictionary to store training history
    H = {"training_loss": [], "val_loss": [], "epochs": []}

    # loop over epochs
    print("[INFO] Training the network...")
    start_time = time.time()

    for epoch in tqdm(range(NUM_EPOCHS)):
        training_loss = 0.0
        val_loss = 0.0
        
        model.train(True)
        for i, (inputs, targets, _) in enumerate(train_loader, 0):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()
                
        avg_training_loss = round(training_loss / len(train_loader), 4)
        
        # update our training history
        H["training_loss"].append(avg_training_loss)
        H["epochs"].append(epoch + 1)
        
        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            for i, (inputs, targets, _) in enumerate(val_loader, 0):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
            avg_val_loss = round(val_loss / len(val_loader), 4)
                
        # update our validation history
        H["val_loss"].append(avg_val_loss)
        print(f'[Epoch {epoch + 1}] Training Loss: {avg_training_loss} Validation Loss: {avg_val_loss}')
    print('Finished Training')

    # display the total time needed to perform the training
    end_time = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(end_time - start_time))
    
    # plot the training loss
    PLOT_PATH = "{} Training Curve.jpg".format(name)
    
    print("Plotting the training loss...")
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["epochs"], H["training_loss"], label="Training Loss")
    plt.plot(H["epochs"], H["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Number of Epochs")
    plt.xticks([i for i in range(0, NUM_EPOCHS + 2, 4)])
    plt.legend(loc="best")
    plt.savefig(os.path.join(BASE_OUTPUT, PLOT_PATH))
    plt.close()

    # serialize the model to disk
    MODEL_PATH = "{}.pth".format(name)
    print(MODEL_PATH)
    torch.save(model, os.path.join(BASE_OUTPUT, MODEL_PATH))
    
def test_model(model, data, scaler, date_time,  ticks, info='Test Dataset'):
    inputs, targets, date_time_index = data.X, data.Y, data.date_time_index
    actual_dates = [date_time[i] for i in date_time_index]
    print(f"{info} | X Shape: {inputs.shape}, Y Shape: {targets.shape}")
    
    with torch.no_grad():
        predictions_all = model(inputs.to(DEVICE)).to('cpu').numpy()

    metrics = []
    plt.figure().set_figwidth(10)
    for i in range(OUTPUT_SIZE):
        predictions = predictions_all[:,:,i]
        target = targets[:,:,i]
        
        if TYPE == 'Univariate':
            prediction_copies = np.repeat(predictions, inputs.shape[1], axis=-1)
            target_copies = np.repeat(target, inputs.shape[1], axis=-1)
        else:
            prediction_copies = np.repeat(predictions, inputs.shape[2], axis=-1)
            target_copies = np.repeat(target, inputs.shape[2], axis=-1)
            
        if i == 0: 
            k = 0    # index for temperature
        else:
            k = 2    # index for pressure
        
        predictions = scaler.inverse_transform(prediction_copies)[:,k]        
        actual_target = scaler.inverse_transform(target_copies)[:,k]

        caption = f"Predictions on {info} "
        file_name = f"of {NAME}.png"
        plt.style.use("ggplot")
        plt.plot(actual_dates[::ticks], actual_target[::ticks], label=f'Actual {OUTPUT_COLS[i]}')
        plt.plot(actual_dates[::ticks], predictions[::ticks], label=f'Predicted {OUTPUT_COLS[i]}')
        plt.title(f'{caption} | {NAME}', fontsize=10)
        plt.xlabel('Date')
        plt.ylabel(OUTPUT_COLS[i])
        plt.xticks(rotation=30)
        plt.legend()
        plt.subplots_adjust(bottom=0.2)
        
        metrics.append(evaluation_metrics(actual_target, predictions, f'{info} - {OUTPUT_COLS[i]} predictions'))
    
    plt.savefig(os.path.join(BASE_OUTPUT, caption + file_name))
    plt.close()
        
    return pd.concat([metric for metric in metrics])


