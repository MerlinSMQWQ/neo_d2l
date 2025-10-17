from ..src.utils.math import linreg, squared_loss
import torch

def test_linreg():
    x = torch.tensor([1])
    w = torch.tensor([2])
    b = torch.tensor([3])
    assert linreg(x, w, b) == torch.tensor([5])
    
def test_squared_loss():
    y_hat = torch.tensor([1, 2, 3])
    y = torch.tensor([2, 3, 4])
    assert squared_loss(y_hat, y).tolist() == [0.5, 0.5, 0.5]