import time
from engine.trainer import train_and_get_accuracy
from optimizers.sa import get_single_neighbor

def tabu_search(train_dataset, val_dataset, init_params, max_iter=10, tabu_size=5, num_neighbors=5):
    current = init_params.copy()
    best = current.copy()
    best_acc = train_and_get_accuracy(current, train_dataset, val_dataset)
    tabu_list = []
    acc_curve = []

    print("\n=== Starting Tabu Search ===")
    for it in range(max_iter):
        neighbors = [get_single_neighbor(current) for _ in range(num_neighbors)]
        best_neighbor, best_neighbor_acc = None, -1.0
        
        for n in neighbors:
            acc = train_and_get_accuracy(n, train_dataset, val_dataset)
            if acc > best_neighbor_acc and n not in tabu_list:
                best_neighbor_acc = acc
                best_neighbor = n

        if best_neighbor is None:
            current = get_single_neighbor(current)
            continue
            
        current = best_neighbor
        tabu_list.append(current)
        if len(tabu_list) > tabu_size: tabu_list.pop(0)

        if best_neighbor_acc > best_acc:
            best_acc = best_neighbor_acc
            best = current.copy()

        acc_curve.append(best_acc)
        print(f"Iteration {it+1}/{max_iter} | Best Accuracy = {best_acc:.4f}")

    return acc_curve, best, best_acc
