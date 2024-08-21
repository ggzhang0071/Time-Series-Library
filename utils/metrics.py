import numpy as np
import torch 
from sklearn.metrics import r2_score


import torch
import numpy as np

def find_lowest_and_highest_points(prediction):
    min_values = []
    min_indices = []
    max_values = []
    max_indices = []
    
    # Iterate over each prediction in the list
    for pred in prediction:
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred)
        # Assuming pred is a tensor of shape (pred_len, 1)

        # Find the minimum value and its index
        min_value, min_index = torch.min(pred, dim=0)
        min_values.append(min_value.item())
        min_indices.append(min_index.item())

        # Find the maximum value and its index
        max_value, max_index = torch.max(pred, dim=0)
        max_values.append(max_value.item())
        max_indices.append(max_index.item())

    return min_values, min_indices, max_values, max_indices



def calculate_accuracy(predictions, targets):
    # Find the lowest and highest points in predictions and targets
    _, pred_min_indices, _, pred_max_indices = find_lowest_and_highest_points(predictions)
    _, target_min_indices, _, target_max_indices = find_lowest_and_highest_points(targets)

    # Convert lists to tensors
    pred_min_indices = torch.tensor(pred_min_indices)
    target_min_indices = torch.tensor(target_min_indices)
    pred_max_indices = torch.tensor(pred_max_indices)
    target_max_indices = torch.tensor(target_max_indices)

    # Calculate the number of correct predictions for lowest points
    correct_min_predictions = torch.sum(pred_min_indices == target_min_indices).item()
    total_min_predictions = len(pred_min_indices)
    min_accuracy = correct_min_predictions / total_min_predictions

    # Calculate the number of correct predictions for highest points
    correct_max_predictions = torch.sum(pred_max_indices == target_max_indices).item()
    total_max_predictions = len(pred_max_indices)
    max_accuracy = correct_max_predictions / total_max_predictions

    return min_accuracy, max_accuracy





def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def R2(pred, true):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - ss_res / ss_tot



def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r2=R2(pred,true)

    return mae, mse, rmse, mape, mspe, r2

if __name__=="__main__":
    # Example usage:
    predictions = [torch.randint(low=0, high=100, size=(10, 1)) for _ in range(100)]

    targets=[torch.randint(low=0, high=100, size=(10, 1)) for _ in range(100)]

    Acc=calculate_accuracy(predictions, targets)


    print(f"ACC:{Acc}")

