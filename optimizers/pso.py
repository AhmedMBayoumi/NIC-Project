import random
import numpy as np
import copy
from engine.trainer import train_and_get_accuracy

def PSO(train_dataset, val_dataset, param_bounds, swarm_size=10, max_iter=20, w=0.5, c1=1.5, c2=1.5):
    particles = []
    for _ in range(swarm_size):
        p = {}
        for key, (min_val, max_val) in param_bounds.items():
            if key in ['num_train_epochs', 'per_device_train_batch_size']:
                p[key] = random.randint(min_val, max_val)
            else:
                p[key] = random.uniform(min_val, max_val)
        particles.append(p)

    velocities = [{k: 0 for k in particles[0]} for _ in range(swarm_size)]
    pbest = copy.deepcopy(particles)
    pbest_scores = [train_and_get_accuracy(p, train_dataset, val_dataset) for p in particles]

    gbest = copy.deepcopy(pbest[np.argmax(pbest_scores)])
    gbest_acc = max(pbest_scores)
    acc_curve = [gbest_acc]

    print("\n=== Starting PSO ===")
    for it in range(max_iter):
        for i, p in enumerate(particles):
            r1, r2 = random.random(), random.random()
            runnable_params = {}
            for key in p:
                min_val, max_val = param_bounds[key]
                velocities[i][key] = (w * velocities[i][key] + c1 * r1 * (pbest[i][key] - p[key]) + c2 * r2 * (gbest[key] - p[key]))
                p[key] += velocities[i][key]

                if key in ['num_train_epochs', 'per_device_train_batch_size']:
                    p[key] = int(np.clip(p[key], min_val, max_val))
                else:
                    p[key] = float(np.clip(p[key], min_val, max_val))
                runnable_params[key] = p[key]

            acc = train_and_get_accuracy(runnable_params, train_dataset, val_dataset)
            if acc > pbest_scores[i]:
                pbest[i] = copy.deepcopy(runnable_params)
                pbest_scores[i] = acc

        best_idx = np.argmax(pbest_scores)
        if pbest_scores[best_idx] > gbest_acc:
            gbest = copy.deepcopy(pbest[best_idx])
            gbest_acc = pbest_scores[best_idx]

        acc_curve.append(gbest_acc)
        print(f"Iteration {it+1}/{max_iter} | Best Accuracy = {gbest_acc:.4f}")

    return acc_curve, gbest, gbest_acc
