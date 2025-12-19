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
from utils.metrics import explain_samples_shap, explain_samples_lime
from torch.utils.data import DataLoader
import shap

def run_pipeline():
    # ... (steps 1-6)
    
    # 7. XAI (SHAP & LIME)
    print("Generating SHAP explanations...")
    tokenizer = get_tokenizer()
    shap_values = explain_samples_shap(final_model, tokenizer, val_dataset, num_samples=3, device='cpu')
    shap.plots.text(shap_values)

    print("Generating LIME explanations...")
    lime_exps = explain_samples_lime(final_model, tokenizer, val_dataset, num_samples=3, device='cpu')
    for i, exp in enumerate(lime_exps):
        exp.show_in_notebook() # Or exp.as_list() for console

if __name__ == "__main__":
    run_pipeline()
