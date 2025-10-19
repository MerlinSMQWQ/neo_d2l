from torch.utils import data
import torch
from typing import Iterable

def load_array(data_arrays: Iterable[torch.Tensor], batch_size: int, is_train: bool=True) -> data.DataLoader:
    """_summary_:
        Load data into DataLoader, 将tonsor加载到DataLoader中  
        
    Args:
        - data_arrays (Iterable[torch.Tensor]): _description_ 数据集  
        - batch_size (int): _description_ 批量大小  
        - is_train (bool, optional): _description_. Defaults to True.  是否用于训练

    Returns:
        data.DataLoader: _description_ DataLoader对象 如果需要用于训练，返回一个乱序的(不训练这无需打乱), 批量大小为batch_size的DataLoader对象
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
