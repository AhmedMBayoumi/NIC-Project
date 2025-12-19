import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from config import config

def ACO(train_features, train_labels, val_features, val_labels, k_features=150):
    n_features = train_features.shape[1]
    n_ants = 20         
    n_iterations = 50
    alpha = 1.0        
    rho = 0.1          
    Q = 1.0            

    def evaluate(feature_mask):
        selected_indices = np.where(feature_mask == 1)[0]
        if len(selected_indices) == 0: return 0.0
            
        X_train_subset = train_features[:, selected_indices]
        X_val_subset = val_features[:, selected_indices]
        
        mlp_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu', solver='adam',
                max_iter=300, early_stopping=True,
                n_iter_no_change=10, random_state=config.RANDOM_STATE
            ))
        ])
        mlp_pipeline.fit(X_train_subset, train_labels)
        preds = mlp_pipeline.predict(X_val_subset)
        return accuracy_score(val_labels, preds)

    pheromone = np.ones(n_features) 
    global_best_accuracy = 0.0
    global_best_features = np.zeros(n_features, dtype=int)

    for iteration in range(n_iterations):
        all_solutions, all_accuracies = [], []

        for ant in range(n_ants):
            probs = pheromone ** alpha
            probs = (probs / np.sum(probs)) if np.sum(probs) > 0 else np.ones(n_features)/n_features

            selected_indices = np.random.choice(n_features, size=k_features, replace=False, p=probs)
            ant_feature_mask = np.zeros(n_features, dtype=int)
            ant_feature_mask[selected_indices] = 1
            
            acc = evaluate(ant_feature_mask)
            all_solutions.append(ant_feature_mask)
            all_accuracies.append(acc)

        pheromone = (1 - rho) * pheromone 
        best_ant_idx = np.argmax(all_accuracies)
        best_features = all_solutions[best_ant_idx]
        best_acc = all_accuracies[best_ant_idx]
        
        pheromone += Q * best_features * best_acc
        
        if best_acc > global_best_accuracy:
            global_best_accuracy = best_acc
            global_best_features = best_features

    return global_best_features, np.where(global_best_features == 1)[0]
