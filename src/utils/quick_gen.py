from torch import normal, matmul
import torch

def synthetic_data(w: torch.Tensor, b: torch.Tensor, num_examples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """_summary_
        Generate data for linear regression y = Xw + b + noise  

    Args:
        w (torch.Tensor): _description_ A weight vector  
        b (torch.Tensor): _description_ A bias vector  
        num_examples (int): _description_ The number of examples  

    Returns:
        tuple[torch.Tensor, torch.Tensor]: _description_ A tuple of (X, Y)  
    """
    X: torch.Tensor = normal(0, 1, (num_examples, len(w)))
    Y: torch.Tensor = matmul(X, w) + b
    Y += normal(0, 0.01, Y.shape)
    return X, Y.reshape((-1, 1))