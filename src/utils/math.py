import math
import numpy as np
import torch
from torch import matmul

def linreg(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """_summary_
        Linear regression, 计算结果  
    
    Args:
        X (torch.Tensor): _description_ 输入向量  
        w (torch.Tensor): _description_ 权重向量  
        b (torch.Tensor): _description_ 偏移量  

    Returns:
        torch.Tensor: _description_ 线性回归结果
    """
    return matmul(X, w) + b