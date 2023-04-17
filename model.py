import torch
from dataset import load_multivariate_dataset, load_temperature_dataset
from config import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from evaluation import save_evaluation_metric
from utility import test_model,  train_model

# Seed
torch.manual_seed(42)

# Define LSTM model
class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1:, :]) 
        
        return out
    
# Initialize model
model = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

if TYPE == "Multivariate":
    train_dataset, train_loader, val_dataset, val_loader, test_dataset , test_loader, \
    scaler, date_time = load_multivariate_dataset(time_steps=TIME_STEPS)
else:
    train_dataset, train_loader, val_dataset, val_loader, test_dataset , test_loader, \
        scaler, date_time = load_temperature_dataset(time_steps=TIME_STEPS)

# Train your model
train_model(model, train_loader, val_loader, criterion, optimizer, name=NAME)

# Load your trained model
model = torch.load(f'output/{NAME}.pth')
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("[INFO] Total Number of Trainable Parameters : {}".format(total_params))

# Test your Model 
train_metric = test_model(model, data=train_dataset, scaler=scaler, date_time=date_time, ticks=500, info="Training Data")
val_metric = test_model(model, data=val_dataset, scaler=scaler, date_time=date_time, ticks=100, info="Validation Data")
test_metric = test_model(model, data=test_dataset, scaler=scaler, date_time=date_time, ticks=100, info="Test Data")

# Save evaluation metrics
print("\nEvaluation Metrics \n")
save_evaluation_metric([train_metric, val_metric, test_metric])