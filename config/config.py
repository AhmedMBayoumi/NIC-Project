import torch
import os

# Paths
DATA_PATH = 'sentiment140/training.1600000.processed.noemoticon.csv' # Note: adjust if data is local
MODEL_NAME = 'bert-base-uncased'

# Data settings
SAMPLE_SIZE = 7000
MAX_LENGTH = 128
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# Optimizer Default Bounds
PARAM_BOUNDS = {
    'learning_rate': (1e-6, 1e-4),
    'num_train_epochs': (2, 5),
    'weight_decay': (0.0, 0.1),
    'warmup_ratio': (0.0, 0.4),
    'adam_beta1': (0.8, 0.99),
    'adam_beta2': (0.9, 0.999),
    'per_device_train_batch_size': (8, 32)
}
