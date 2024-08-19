import numpy as np
import torch 
from sklearn.metrics import r2_score


def find_lowest_point(prediction):
    min_values = []
    min_indices = []
    
    # Iterate over each prediction in the list
    for pred in prediction:
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred)
        # Assuming pred is a tensor of shape (pred_len, 1)

        min_value, min_index = torch.min(pred, dim=0)  # Find the minimum along the time dimension
        min_values.append(min_value.item())  # Convert tensor to a Python scalar and store
        min_indices.append(min_index.item())  # Store the index as a Python scalar

    return min_values, min_indices
def calculate_accuracy(predictions, targets):
    # Find the lowest points in predictions and targets
    _, pred_min_indices = find_lowest_point(predictions)
    _, target_min_indices = find_lowest_point(targets)

    # Convert lists to tensors
    pred_min_indices = torch.tensor(pred_min_indices)
    target_min_indices = torch.tensor(target_min_indices)

    # Calculate the number of correct predictions
    correct_predictions = torch.sum(pred_min_indices == target_min_indices).item()
    total_predictions = len(pred_min_indices)

    accuracy = correct_predictions / total_predictions
    return accuracy


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

def R2_score_adjust(true,pred):
    # 将 true 和 pred 转换成 numpy 数组
    true = np.array(true)  # (95, 5, 1)
    pred = np.array(pred)  # (95, 5, 1)

    # 扁平化为 (95 * 5, 1) 的数组
    true_flat = true.reshape(-1)
    pred_flat = pred.reshape(-1)
    r21=r2_score(true_flat, pred_flat)
    return r21

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r2=R2(pred,true)
    r21 = R2_score_adjust(true,pred)


    return mae, mse, rmse, mape, mspe,r2, r21

if __name__=="__main__":
    # Example usage:
    predictions = [torch.randint(low=0, high=100, size=(10, 1)) for _ in range(100)]

    targets=[torch.randint(low=0, high=100, size=(10, 1)) for _ in range(100)]

    Acc=calculate_accuracy(predictions, targets)


    print(f"ACC:{Acc}")

