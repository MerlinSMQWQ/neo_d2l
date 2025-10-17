from typing import Iterable
import torch

def sgd(params: Iterable[torch.Tensor], lr: float, batch_size: int):
    """_summary_:
        SGD, 随机梯度下降

    Args:
        params (Iterable[torch.Tensor]): _description_ 参数  
        lr (float): _description_ 学习率  
        batch_size (int): _description_ 批量大小  
    """
    with torch.no_grad():
        for param in params:
            if param.grad is not None:
                param.grad -= lr * param.grad / batch_size
                # 计算完以后需要将梯度归零
                param.grad.zero_()