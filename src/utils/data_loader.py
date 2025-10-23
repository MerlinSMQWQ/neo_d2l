import torchvision
from numpy import ndarray
from torch.utils import data
import torch
from typing import Iterable
from torchvision import transforms
from matplotlib import pyplot as plt
from matplotlib import axes as Axes
import sys

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

def get_fashion_mnist_labels(labels: Iterable[int]) -> list[str]:
    """_summary_:
        

    Args:
        labels (Iterable): _description_

    Returns:
        list[str]: _description_
    """
    text_labels: list[str] = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

    return [text_labels[int(i)] for i in labels]


def get_dataloader_workers(num_workers: int = 4):  #@save
    """_summary_
        在非Windows的平台上, 使用4个进程来读取数据
    """
    return 0 if sys.platform.startswith('win') else num_workers

