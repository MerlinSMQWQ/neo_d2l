from ..src.utils.quick_gen import synthetic_data
import torch


def test_synthetic_data():
    w = torch.tensor([2.0, -3.4])
    b = torch.tensor([4.2])
    X, y = synthetic_data(w, b, 1000)
    print(X.shape, y.shape)