import random
import numpy as np
import copy
from engine.trainer import train_and_get_accuracy

class Wolf:
    def __init__(self, bounds):
        self.position = {}
        self.dim = len(bounds)
        self.param_keys = list(bounds.keys())
        for param, (min_val, max_val) in bounds.items():
            self.position[param] = random.uniform(min_val, max_val)
        self.fitness = float('inf')

    def get_runnable_params(self):
        params = copy.deepcopy(self.position)
        if 'num_train_epochs' in params:
            params['num_train_epochs'] = int(round(params['num_train_epochs']))
        if 'per_device_train_batch_size' in params:
            params['per_device_train_batch_size'] = int(round(params['per_device_train_batch_size']))
        return params

    def get_pos_numpy(self):
        return np.array([self.position[k] for k in self.param_keys])

    def set_pos_numpy(self, pos_array, bounds):
        for i, key in enumerate(self.param_keys):
            min_val, max_val = bounds[key]
            self.position[key] = np.clip(pos_array[i], min_val, max_val)

def gwo(bounds, train_dataset, val_dataset, n_wolves=5, max_iter=10):
    pack = [Wolf(bounds) for _ in range(n_wolves)]
    Alpha_wolf = Wolf(bounds); Alpha_wolf.fitness = float('inf')
    Beta_wolf = Wolf(bounds); Beta_wolf.fitness = float('inf')
    Delta_wolf = Wolf(bounds); Delta_wolf.fitness = float('inf')
    
    cost_history = []

    for wolf in pack:
        wolf.fitness = 1.0 - train_and_get_accuracy(wolf.get_runnable_params(), train_dataset, val_dataset)
        if wolf.fitness < Alpha_wolf.fitness:
            Delta_wolf = copy.deepcopy(Beta_wolf)
            Beta_wolf = copy.deepcopy(Alpha_wolf)
            Alpha_wolf = copy.deepcopy(wolf)
        elif wolf.fitness < Beta_wolf.fitness:
            Delta_wolf = copy.deepcopy(Beta_wolf)
            Beta_wolf = copy.deepcopy(wolf)
        elif wolf.fitness < Delta_wolf.fitness:
            Delta_wolf = copy.deepcopy(wolf)

    for iter in range(max_iter):
        a = 2 - iter * (2 / max_iter)
        X_alpha, X_beta, X_delta = Alpha_wolf.get_pos_numpy(), Beta_wolf.get_pos_numpy(), Delta_wolf.get_pos_numpy()

        for wolf in pack:
            X_current = wolf.get_pos_numpy()
            r1, r2 = np.random.rand(wolf.dim), np.random.rand(wolf.dim)
            A1, C1 = 2 * a * r1 - a, 2 * r2
            D_alpha = np.abs(C1 * X_alpha - X_current)
            X1 = X_alpha - A1 * D_alpha
            
            r1, r2 = np.random.rand(wolf.dim), np.random.rand(wolf.dim)
            A2, C2 = 2 * a * r1 - a, 2 * r2
            D_beta = np.abs(C2 * X_beta - X_current)
            X2 = X_beta - A2 * D_beta

            r1, r2 = np.random.rand(wolf.dim), np.random.rand(wolf.dim)
            A3, C3 = 2 * a * r1 - a, 2 * r2
            D_delta = np.abs(C3 * X_delta - X_current)
            X3 = X_delta - A3 * D_delta

            X_new = (X1 + X2 + X3) / 3
            temp_wolf = Wolf(bounds)
            temp_wolf.set_pos_numpy(X_new, bounds)
            new_fitness = 1.0 - train_and_get_accuracy(temp_wolf.get_runnable_params(), train_dataset, val_dataset)
            
            if new_fitness < wolf.fitness:
                wolf.set_pos_numpy(X_new, bounds)
                wolf.fitness = new_fitness

        for wolf in pack:
            if wolf.fitness < Alpha_wolf.fitness:
                Delta_wolf = copy.deepcopy(Beta_wolf); Beta_wolf = copy.deepcopy(Alpha_wolf); Alpha_wolf = copy.deepcopy(wolf)
            elif wolf.fitness < Beta_wolf.fitness:
                Delta_wolf = copy.deepcopy(Beta_wolf); Beta_wolf = copy.deepcopy(wolf)
            elif wolf.fitness < Delta_wolf.fitness:
                Delta_wolf = copy.deepcopy(wolf)
        
        cost_history.append(Alpha_wolf.fitness)
        print(f"Iteration {iter+1}/{max_iter} | Best Accuracy: {1.0 - Alpha_wolf.fitness:.4f}")

    return Alpha_wolf.get_runnable_params(), cost_history
