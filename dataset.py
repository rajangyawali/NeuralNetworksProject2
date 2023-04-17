import torch
from torch.utils.data import DataLoader, Dataset
from config import *
import pandas as pd
import numpy as np
from copy import deepcopy as dc
from sklearn.preprocessing import StandardScaler
import math

# Climate Dataset Custom Class
class ClimateDataset(Dataset):
    def __init__(self, X, Y, date_time_index):
        self.X = X
        self.Y = Y
        self.date_time_index = date_time_index


    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.date_time_index[i]
    
def read_data():
    df = pd.read_csv('data/jena_climate_2009_2016.csv')
    
    # Extracting records hour-wise
    df = df[5::6]
    df = df.reset_index(drop=True)
    
    # Converting string datetime to pandas datetime
    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    
    # Setting index to Date Time
    df.index = df['Date Time']
    
    return df
    

def create_temperature_data_for_lstm(data, t_past, t_future):
    X, Y = [], []
    for i in range(t_past, len(data) - t_future + 1):
        X.append(data[i - t_past : i, 0 : data.shape[1]])
        Y.append(data[i + t_future - 1 : i + t_future, [0]])
        
    return np.array(X), np.array(Y)

# Use this function if you need to output multiple values
def create_multivariate_data_for_lstm(data, t_past, t_future):
    X, Y = [], []
    cols_index = [0, 2] # index of output columns
    for i in range(t_past, len(data) - t_future + 1):
        X.append(data[i - t_past : i, 0 : data.shape[1]])
        Y.append(data[i + t_future - 1 : i + t_future, cols_index])
        
    return np.array(X), np.array(Y)

def create_pytorch_dataset(X, Y, date_time_index):
    
    # Let's create train, validation and test split
    train_val_test_split = [math.floor(len(X)*0.7), math.floor(len(X)*0.85)]
    train_X, train_Y, train_date_time_index = X[:train_val_test_split[0]], Y[:train_val_test_split[0]], date_time_index[:train_val_test_split[0]]
    val_X, val_Y, val_date_time_index = X[train_val_test_split[0]:train_val_test_split[1]], Y[train_val_test_split[0]:train_val_test_split[1]], date_time_index[train_val_test_split[0]:train_val_test_split[1]]
    test_X, test_Y, test_date_time_index = X[train_val_test_split[1]:], Y[train_val_test_split[1]:], date_time_index[train_val_test_split[1]:]
    
    # Let's convert data to pytorch tensors
    train_X, val_X, test_X  = torch.Tensor(train_X), torch.Tensor(val_X), torch.Tensor(test_X)
    train_Y, val_Y, test_Y = torch.Tensor(train_Y), torch.Tensor(val_Y), torch.Tensor(test_Y)
    
    train_dataset = ClimateDataset(train_X, train_Y, train_date_time_index)
    val_dataset = ClimateDataset(val_X, val_Y, val_date_time_index)
    test_dataset = ClimateDataset(test_X, test_Y, test_date_time_index)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_dataset, train_loader, val_dataset, val_loader, test_dataset , test_loader



def load_temperature_dataset(time_steps):
    # Read data
    df = read_data()
    
    # Creating a custom dataset with Date Time and Temperature Only
    df_univariate = df.loc[:, ['T (degC)']]
    df_univariate.to_csv('temperature_prediction_univariate_data.csv')
    
    # Let's do some normalization
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df_univariate)
    print("Normalized Data Shape:", normalized_data.shape)
    # np.savetxt('normalized_univariate.csv', normalized_data)
    
    # Let's create data ready to be used by LSTM
    X, Y = create_temperature_data_for_lstm(normalized_data, time_steps, 1)
    date_time =  df.index
    date_time_index = [i + time_steps for i in range(len(X))]
    print(f"X shape: {X.shape} | Y Shape: {Y.shape}")
        
    train_ds, train_loader, val_ds, val_loader, test_ds , test_loader = create_pytorch_dataset(X, Y, date_time_index)
    
    return train_ds, train_loader, val_ds, val_loader, test_ds , test_loader, scaler, date_time

def load_multivariate_dataset(time_steps):
    # Read data
    df = read_data()
    
    # Creating a custom dataset with Date Time, Pressure, Temperature, Dew Point Temperature 
    # and Relative  Humidity Only
    df_multivariate = df.loc[:, ['T (degC)', 'Tdew (degC)', 'p (mbar)',	'rh (%)']]
    df_multivariate.to_csv('temperature_prediction_multivariate_data.csv')
    
    # Let's do some normalization
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df_multivariate)
    print("Normalized Data Shape:", normalized_data.shape)
    # np.savetxt('normalized_multivariate.csv', normalized_data)
    
    # Let's create data ready to be used by LSTM
    if OUTPUT_SIZE != 1:
        X, Y = create_multivariate_data_for_lstm(normalized_data, time_steps, 1)
    else:
        X, Y = create_temperature_data_for_lstm(normalized_data, time_steps, 1)
    date_time =  df.index
    date_time_index = [i + time_steps for i in range(len(X))]
    print(f"X shape: {X.shape} | Y Shape: {Y.shape}")
    
    train_ds, train_loader, val_ds, val_loader, test_ds , test_loader = create_pytorch_dataset(X, Y, date_time_index)
    
    return train_ds, train_loader, val_ds, val_loader, test_ds , test_loader, scaler, date_time
