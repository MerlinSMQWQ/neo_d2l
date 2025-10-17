import math
import numpy as np
import torch
from torch import matmul

reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)

def linreg(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """_summary_:
        Linear regression, 计算结果  
    
    Args:
        X (torch.Tensor): _description_ 输入向量  
        w (torch.Tensor): _description_ 权重向量  
        b (torch.Tensor): _description_ 偏移量  

    Returns:
        torch.Tensor: _description_ 线性回归结果
    """
    return matmul(X, w) + b

def squared_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """_summary_:
        Squared loss, 平方损失函数  

    Args:
        y_hat (torch.Tensor): _description_ 预测值  
        y (torch.Tensor): _description_ 真实值  

    Returns:
        torch.Tensor: _description_ 损失值  
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2