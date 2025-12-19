import torch
from config import config
from utils.helpers import set_seed, get_device
from data.data_loader import load_and_prep_data
from models.model_utils import get_classification_model, get_tokenizer, get_base_model, extract_features
from engine.trainer import train_and_get_accuracy
from optimizers.aco import ACO
from optimizers.sa import simulated_annealing
from optimizers.gwo import gwo
from optimizers.tabu import tabu_search
from optimizers.pso import PSO
from optimizers.woa_ga import genetic_optimizer
from optimizers.firefly_memetic import firefly_optimize, memetic_search
from utils.metrics import explain_samples
from torch.utils.data import DataLoader
import shap

def run_pipeline():
    # 1. Setup
    set_seed(config.RANDOM_STATE)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_dataset, val_dataset = load_and_prep_data()
    if train_dataset is None: return

    # 3. Model & Feature Extraction (Optional, for ACO)
    print("Extracting features for ACO...")
    model_base = get_base_model()
    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    train_features, train_labels = extract_features(model_base, train_loader, device)
    val_features, val_labels = extract_features(model_base, val_loader, device)

    # 4. Feature Selection (ACO example)
    print("Running ACO for feature selection...")
    best_mask, best_indices = ACO(train_features, train_labels, val_features, val_labels, k_features=150)
    print(f"Selected {len(best_indices)} features.")

    # 5. Hyperparameter Optimization (Example: Simulated Annealing)
    print("Running HPO with Simulated Annealing...")
    initial_params = {
        'learning_rate': 3e-5,
        'num_train_epochs': 2,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'per_device_train_batch_size': 16
    }
    
    # Passing datasets to SA for HPO
    trace, best_hparams, best_cost = simulated_annealing(initial_params, train_dataset, val_dataset, max_iter=5)
    print(f"Best HPO Accuracy: {1.0 - best_cost:.4f}")

    # 6. Final Training with Optimized Parameters
    print("Training final model...")
    final_model = get_classification_model()
    # Note: Use best_hparams here in a full training run
    # For now, we reuse the training function or call Trainer directly for full epochs
    # For brevity in this template, we show the HPO result as the end of this example flow.

    # 7. XAI (SHAP)
    print("Generating SHAP explanations...")
    tokenizer = get_tokenizer()
    shap_values = explain_samples(final_model, tokenizer, val_dataset, num_samples=3, device='cpu')
    shap.plots.text(shap_values)

if __name__ == "__main__":
    run_pipeline()
