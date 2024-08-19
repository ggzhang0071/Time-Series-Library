# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape1_loss(nn.Module):
    def __init__(self):
        super(mape1_loss, self).__init__()  # 修改这里，使用正确的类名

    def forward(self, forecast: t.Tensor, target: t.Tensor) -> t.float:  # 添加 self 参数
        # 确保实际值中没有零值，避免除零错误
        # 使用 clamp 方法将所有实际值至少设为非常小的正数
        epsilon = 1e-8  # 防止除零
        actual_safe = t.clamp(target, min=epsilon)
        loss = t.abs((actual_safe - forecast) / actual_safe)
        #print(f"loss is {loss}")
        return t.mean(loss) * 100  # 结果乘以 100 转换为百分比

class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor=None) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
    

        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)
    

class r2_loss(nn.Module):
    def __init__(self):
        super(r2_loss, self).__init__()
        
    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor = None) -> t.float:
        
        """
        R-Squared loss function.

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value (1 - R-Squared)
        """
        # Apply mask to forecast and target
        if mask is None:
            masked_forecast=forecast
            masked_target=target
        else:

            masked_forecast = forecast * mask
            masked_target = target * mask

        # Compute mean of the target
        target_mean = t.mean(masked_target, dim=1, keepdim=True)

        # Compute the total sum of squares (TSS) and the residual sum of squares (RSS)
        tss = t.sum((masked_target - target_mean) ** 2, dim=1)
        rss = t.sum((masked_target - masked_forecast) ** 2, dim=1)

        # Compute R-Squared
        r2 = 1 - divide_no_nan(rss, tss)

        # Take mean across batches and return (1 - R-Squared)
        return t.mean(1 - r2)
    
if __name__=="__main__":
    # 示例数据
    forecast = t.tensor([[110.0, 120.0, 130.0], [210.0, 220.0, 230.0]])
    target = t.tensor([[100.0, 120.0, 0.0], [200.0, 0.0, 230.0]])
    mask = t.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])  # 忽略了第三个时间步长和第二个batch的第二个时间步长

    # 初始化并计算loss
    criterion = r2_loss()
    loss = criterion(None, 1, forecast, target)

    print(loss)
