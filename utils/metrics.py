import numpy as np
from sklearn.metrics import r2_score


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
