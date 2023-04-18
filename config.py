import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

BASE_OUTPUT = "output"



BATCH_SIZE = 2048
NUM_WORKERS = 2


NUM_EPOCHS = 15
LEARNING_RATE = 0.001
LOSS = "MSE"
OPTIMIZER = "Adam"
TYPE = "Multivariate"

if TYPE == "Univariate":
    INPUT_SIZE = 1
elif TYPE == "Multivariate":
    INPUT_SIZE = 4


HIDDEN_SIZE = 10
NUM_LAYERS = 2
TIME_STEPS = 24 * 7

# OUTPUT_COLS = ['T (degC)']
OUTPUT_COLS = ['T (degC)', 'p (mbar)']
OUTPUT_SIZE = len(OUTPUT_COLS)


NAME = f"{TYPE} LSTM with {LOSS}, {OPTIMIZER}, {NUM_EPOCHS} epochs, {TIME_STEPS} time steps, {LEARNING_RATE} LR, {INPUT_SIZE} ip, {OUTPUT_SIZE} op"
