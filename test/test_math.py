from ..src.utils.math import linreg
import torch

def test_linreg():
    x = torch.tensor([1])
    w = torch.tensor([2])
    b = torch.tensor([3])
    assert linreg(x, w, b) == torch.tensor([5])