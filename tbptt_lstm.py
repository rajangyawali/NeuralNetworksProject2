import time
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
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
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True).to(DEVICE)
        
        self.fc = nn.Linear(self.hidden_size, self.output_size).to(DEVICE)

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
        
        optimizer.zero_grad()
        for j in range(0, TIME_STEPS, 1):
            
            inputs_seq = inputs[:, j:j+TIME_STEPS, :]
            outputs = model(inputs_seq)
            loss = criterion(outputs, targets)
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
PLOT_PATH = "{} Training Curve.jpg".format(NAME)

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
MODEL_PATH = "{}.pth".format(NAME)
print(MODEL_PATH)
torch.save(model, os.path.join(BASE_OUTPUT, MODEL_PATH))

# Load your trained model
model = torch.load(f'output/{NAME}.pth')
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("[INFO] Total Number of Trainable Parameters : {}".format(total_params))

# Test your Model 
train_metric = test_model(model, data=train_dataset, scaler=scaler, date_time=date_time, ticks=48, info="Training Data")
val_metric = test_model(model, data=val_dataset, scaler=scaler, date_time=date_time, ticks=48, info="Validation Data")
test_metric = test_model(model, data=test_dataset, scaler=scaler, date_time=date_time, ticks=48, info="Test Data")

# Save evaluation metrics
print("\nEvaluation Metrics \n")
save_evaluation_metric([train_metric, val_metric, test_metric])