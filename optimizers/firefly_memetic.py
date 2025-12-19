import random
import numpy as np
import copy
import matplotlib.pyplot as plt
from engine.trainer import train_and_get_accuracy

def random_params(bounds):
    params = {}
    for k, (low, high) in bounds.items():
        if k in ['num_train_epochs', 'per_device_train_batch_size']:
            params[k] = int(random.uniform(low, high))
        else:
            params[k] = float(random.uniform(low, high))
    return params

def clip_params(params, bounds):
    new_p = {}
    for k, v in params.items():
        low, high = bounds[k]
        if k in ['num_train_epochs', 'per_device_train_batch_size']:
            new_p[k] = int(np.clip(v, low, high))
        else:
            new_p[k] = float(np.clip(v, low, high))
    return new_p

def firefly_distance(p1, p2, keys):
    v1 = np.array([p1[k] for k in keys], dtype=float)
    v2 = np.array([p2[k] for k in keys], dtype=float)
    return np.linalg.norm(v1 - v2)

def firefly_optimize(train_dataset, val_dataset, bounds, n_fireflies=3, n_iter=3, beta0=1.0, gamma=1.0, alpha=0.2):
    keys = list(bounds.keys())
    fireflies = [random_params(bounds) for _ in range(n_fireflies)]
    scores = [train_and_get_accuracy(p, train_dataset, val_dataset) for p in fireflies]

    best_acc = max(scores)
    best_params = copy.deepcopy(fireflies[int(np.argmax(scores))])
    best_history = [best_acc]

    for it in range(n_iter):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if scores[j] > scores[i]:
                    d = firefly_distance(fireflies[i], fireflies[j], keys)
                    beta = beta0 * np.exp(-gamma * d * d)
                    new_p = copy.deepcopy(fireflies[i])
                    for k in keys:
                        new_p[k] = new_p[k] + beta * (fireflies[j][k] - fireflies[i][k]) + alpha * (random.random() - 0.5)
                    new_p = clip_params(new_p, bounds)
                    new_acc = train_and_get_accuracy(new_p, train_dataset, val_dataset)
                    if new_acc > scores[i]:
                        fireflies[i], scores[i] = new_p, new_acc
        best_acc = max(scores)
        best_params = copy.deepcopy(fireflies[int(np.argmax(scores))])
        best_history.append(best_acc)
    return best_params, best_acc, best_history

def small_change(params, bounds, scale=0.1):
    new_p = copy.deepcopy(params)
    key = random.choice(list(new_p.keys()))
    low, high = bounds[key]
    if key in ['num_train_epochs', 'per_device_train_batch_size']:
        step = max(1, int((high - low) * scale))
        new_p[key] += random.choice([-step, step])
    else:
        new_p[key] += random.uniform(-(high-low)*scale, (high-low)*scale)
    return clip_params(new_p, bounds)

def memetic_search(train_dataset, val_dataset, start_params, bounds, n_steps=3):
    best_p = copy.deepcopy(start_params)
    best_acc = train_and_get_accuracy(best_p, train_dataset, val_dataset)
    history = [best_acc]

    for s in range(n_steps):
        cand = small_change(best_p, bounds)
        cand_acc = train_and_get_accuracy(cand, train_dataset, val_dataset)
        if cand_acc > best_acc:
            best_acc, best_p = cand_acc, cand
        history.append(best_acc)
    return best_p, best_acc, history

def memetic_optimization_for_firefly(train_dataset, val_dataset, model_bounds, n_steps=3):
    """
    Step 3 Meta-Optimization:
    Use Memetic Search to optimize the hyperparameters of the Firefly Algorithm.
    The Firefly Algorithm then optimizes the BERT model.
    """
    firefly_param_ranges = {
        'beta0': (0.5, 2.0),
        'gamma': (0.1, 5.0),
        'alpha': (0.1, 0.5)
    }
    
    def get_firefly_config_accuracy(config):
        # Run a small firefly optimization with these params
        _, acc, _ = firefly_optimize(
            train_dataset, val_dataset, model_bounds, 
            n_fireflies=3, n_iter=2, 
            beta0=config['beta0'], gamma=config['gamma'], alpha=config['alpha']
        )
        return acc

    # Initial firefly config
    current_config = {
        'beta0': 1.0,
        'gamma': 1.0,
        'alpha': 0.2
    }
    best_acc = get_firefly_config_accuracy(current_config)
    
    for s in range(n_steps):
        # Small change to firefly config
        key = random.choice(list(firefly_param_ranges.keys()))
        low, high = firefly_param_ranges[key]
        candidate_config = copy.deepcopy(current_config)
        candidate_config[key] = np.clip(candidate_config[key] + random.uniform(-0.1, 0.1), low, high)
        
        cand_acc = get_firefly_config_accuracy(candidate_config)
        if cand_acc > best_acc:
            best_acc, current_config = cand_acc, candidate_config
            
    return current_config, best_acc
