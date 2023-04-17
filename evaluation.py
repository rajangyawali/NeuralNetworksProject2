import os
from config import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table


def evaluate_mape(y_true, y_pred):
    epsilon = 1e-03
    sum = 0
    for i in range(len(y_pred)):
        if y_true[i] >= -epsilon and y_true[i] < epsilon:
            sum += np.abs((y_true[i] - y_pred[i] + epsilon)/(y_true[i] + epsilon))
        else:
            sum += np.abs((y_true[i] - y_pred[i])/(y_true[i]))
    return round(sum/len(y_true) * 100, 4)

def evaluation_metrics(y_true, y_pred, info):
    
    # calculate the MAE    
    mae = round(np.mean(np.abs(y_true - y_pred)), 4)

    # calculate the RMSE
    rmse = round(np.sqrt(np.mean(np.square(y_true - y_pred))), 4)

    # calculate the MAPE
    mape = evaluate_mape(y_true, y_pred)

    # calculate the SMAPE
    smape = round(np.mean(np.abs((y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))) * 100, 4)

    # calculate the MDA
    mda = round(np.mean(np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])) * 100, 4)
    
    df = pd.DataFrame(data=np.array([[info, mae, rmse, mape, smape, mda]]),
                      columns=['Dataset', 'MAE', 'RMSE', 'MAPE', 'SMAPE', 'MDA'],
                      )
    
    return df

def save_evaluation_metric(metrics):
    df = pd.concat([metric for metric in metrics])
    df = df.set_index('Dataset')
    print(df)
    

    _, ax = plt.subplots(figsize=(10,4))
    ax.axis('off')
    ax.set_frame_on(False)
    ax.set_title(f'LSTM Evaluation Metrics | {NAME}', fontsize=10)
    t = table(ax, df, loc='center', cellLoc='center', rowLoc='center', fontsize=14, cellColours=[['lightgray']*len(df.columns)]*len(df.index))
    for k in range(5):
        t.auto_set_column_width(k)
    file_name = f"Evaluation Metrics  of {NAME}.png"
    plt.savefig(os.path.join(BASE_OUTPUT, file_name))