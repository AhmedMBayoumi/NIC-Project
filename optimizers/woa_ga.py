import random
import numpy as np
import copy
from engine.trainer import train_and_get_accuracy

class WhaleOptimizationAlgorithm:
    def __init__(self, train_dataset, val_dataset, bounds, n_whales=5, max_iter=10):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.bounds = bounds
        self.param_keys = list(bounds.keys())
        self.lb = np.array([bounds[k][0] for k in self.param_keys])
        self.ub = np.array([bounds[k][1] for k in self.param_keys])
        self.dim = len(self.param_keys)
        self.whales = np.random.uniform(low=self.lb, high=self.ub, size=(n_whales, self.dim))
        self.best_whale = None
        self.best_score = float('inf')
        self.convergence_curve = []

    def _get_params_from_vector(self, vector):
        params = {}
        for i, key in enumerate(self.param_keys):
            val = np.clip(vector[i], self.lb[i], self.ub[i])
            if key in ['num_train_epochs', 'per_device_train_batch_size']:
                params[key] = int(round(val))
            else:
                params[key] = float(val)
        return params

    def fitness(self, whale):
        params = self._get_params_from_vector(whale)
        return 1.0 - train_and_get_accuracy(params, self.train_dataset, self.val_dataset)

    def update_position(self, whale, a, b, c, best_whale_position, n_whales):
        r1, r2 = np.random.random(), np.random.random()
        A, C = 2 * a * r1 - a, 2 * r2
        p = np.random.random()
        
        if p < c:
            if abs(A) < 1:
                D = np.abs(C * best_whale_position - whale)
                new_position = best_whale_position - A * D
            else:
                rand_idx = np.random.randint(0, n_whales)
                rand_whale = self.whales[rand_idx]
                D = np.abs(C * rand_whale - whale)
                new_position = rand_whale - A * D
        else:
            D_prime = np.abs(best_whale_position - whale)
            l = np.random.uniform(-1, 1)
            new_position = D_prime * np.exp(b * l) * np.cos(2 * np.pi * l) + best_whale_position
        return new_position

    def optimize(self, a_start=2.0, b_val=1.0, c_prob=0.5):
        n_whales = len(self.whales)
        for i in range(n_whales):
            fit = self.fitness(self.whales[i])
            if fit < self.best_score:
                self.best_score = fit
                self.best_whale = self.whales[i].copy()

        for t in range(10): # Example max_iter
            a = a_start - t * (a_start / 10)
            for i in range(n_whales):
                self.whales[i] = self.update_position(self.whales[i], a, b_val, c_prob, self.best_whale, n_whales)
                self.whales[i] = np.clip(self.whales[i], self.lb, self.ub)
                fitness_value = self.fitness(self.whales[i])
                if fitness_value < self.best_score:
                    self.best_score = fitness_value
                    self.best_whale = self.whales[i].copy()
            self.convergence_curve.append(self.best_score)
        return self.best_whale, self.best_score, self.convergence_curve

def genetic_optimizer(train_dataset, val_dataset, whale_bounds, ga_pop_size=5, ga_generations=3):
    param_ranges = [(1.0, 3.0), (0.1, 2.0), (0.1, 0.9)] # a_start, b, c
    population = [[random.uniform(low, high) for low, high in param_ranges] for _ in range(ga_pop_size)]
    best_ga_individual, best_ga_fitness = None, float('inf')
    history = []

    for gen in range(ga_generations):
        fitnesses = []
        for i, ind in enumerate(population):
            a_start, b, c = ind
            woa = WhaleOptimizationAlgorithm(train_dataset, val_dataset, whale_bounds, n_whales=3, max_iter=3)
            _, score, _ = woa.optimize(a_start=a_start, b_val=b, c_prob=c)
            fitnesses.append(score)
            if score < best_ga_fitness:
                best_ga_fitness = score
                best_ga_individual = ind

        history.append(1.0 - best_ga_fitness)
        selected = []
        for _ in range(ga_pop_size):
            i1, i2 = random.randint(0, ga_pop_size-1), random.randint(0, ga_pop_size-1)
            selected.append(population[i1] if fitnesses[i1] < fitnesses[i2] else population[i2])
        
        next_gen = []
        for i in range(0, ga_pop_size, 2):
            p1, p2 = selected[i], selected[i+1] if i+1 < ga_pop_size else selected[0]
            if random.random() < 0.7:
                split = random.randint(1, 2)
                child1, child2 = p1[:split] + p2[split:], p2[:split] + p1[split:]
            else:
                child1, child2 = p1[:], p2[:]
            
            for child in [child1, child2]:
                if random.random() < 0.2:
                    m_idx = random.randint(0, 2)
                    child[m_idx] = random.uniform(param_ranges[m_idx][0], param_ranges[m_idx][1])
            next_gen.extend([child1, child2])
        population = next_gen[:ga_pop_size]

    return best_ga_individual, history
