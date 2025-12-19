import random
import numpy as np
import copy
from engine.trainer import train_and_get_accuracy

def get_single_neighbor(current_state):
    new_state = copy.deepcopy(current_state)
    param_to_change = random.choice(list(new_state.keys()))
    
    if param_to_change == 'learning_rate':
        new_state['learning_rate'] *= (10 ** random.uniform(-0.1, 0.1))
        new_state['learning_rate'] = max(1e-6, min(1e-3, new_state['learning_rate']))
    elif param_to_change == 'num_train_epochs':
        new_state['num_train_epochs'] += random.choice([-1, 1])
        new_state['num_train_epochs'] = max(1, min(4, new_state['num_train_epochs']))
    elif param_to_change == 'weight_decay':
        new_state['weight_decay'] += random.uniform(-0.01, 0.01)
        new_state['weight_decay'] = max(0, min(0.3, new_state['weight_decay']))
    elif param_to_change == 'warmup_ratio':
        new_state['warmup_ratio'] += random.uniform(-0.05, 0.05)
        new_state['warmup_ratio'] = max(0.0, min(0.4, new_state['warmup_ratio']))
    elif param_to_change == 'adam_beta1':
        new_state['adam_beta1'] += random.uniform(-0.01, 0.01)
        new_state['adam_beta1'] = max(0.8, min(0.99, new_state['adam_beta1']))
    elif param_to_change == 'adam_beta2':
        new_state['adam_beta2'] += random.uniform(-0.001, 0.001)
        new_state['adam_beta2'] = max(0.9, min(0.999, new_state['adam_beta2']))
    elif param_to_change == 'per_device_train_batch_size':
        new_state['per_device_train_batch_size'] += random.choice([-2, 2, -4, 4])
        new_state['per_device_train_batch_size'] = max(8, min(32, new_state['per_device_train_batch_size']))
        
    return new_state

def generate_neighbors(current_state, n=5):
    neighbors = []
    for _ in range(n):
        neighbors.append(get_single_neighbor(current_state))
    return neighbors

def simulated_annealing(initial, train_dataset, val_dataset, T=1.0, alpha=0.9, max_iter=20):
    current = initial
    current_acc = train_and_get_accuracy(current, train_dataset, val_dataset)
    current_cost = 1.0 - current_acc
    
    best_ever_state = current
    best_ever_cost = current_cost
    trace = [(current, current_cost)]

    for i in range(max_iter):
        T = T * alpha  
        neighbor = random.choice(generate_neighbors(current))
        neighbor_acc = train_and_get_accuracy(neighbor, train_dataset, val_dataset)
        neighbor_cost = 1.0 - neighbor_acc

        if neighbor_cost < best_ever_cost:
            best_ever_state, best_ever_cost = neighbor, neighbor_cost

        cost_diff = neighbor_cost - current_cost
        if cost_diff < 0 or random.random() < np.exp(-cost_diff / T):
            current, current_cost = neighbor, neighbor_cost

        trace.append((current, current_cost))
        print(f"Iteration {i+1}/{max_iter} | Best Accuracy={1.0-best_ever_cost:.4f}")
        
    return trace, best_ever_state, best_ever_cost
