from typing import Iterable
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.scale import ScaleBase
import numpy as np

# 
def use_TKAgg_display() -> None:
    """_summary_:
        使用svg, 运行直接显示, 不依赖jupyter
    """
    matplotlib.use('TKAgg')
    plt.ion

def set_figsize(figsize: tuple[float, float] = (3.5, 2.5)) -> None:
    """_summary_:
        设置图像大小

    Args:
        - figsize (tuple[float, float], optional): _description_. Defaults to (3.5, 2.5).
    """
    use_TKAgg_display()
    plt.rcParams['figure.figsize'] = figsize
    
def set_axes(axes: ax.Axes, xlabel: str, ylabel: str, xlim: tuple[float, float]|None, 
             ylim: tuple[float, float]|None, xscale: str | ScaleBase, yscale: str | ScaleBase, legend: Iterable[str]|None) -> None:
    """_summary_:
        设置图像坐标轴
    Args:
        - axes (ax.Axes): _description_. 坐标轴  
        - xlabel (str): _description_. x轴标签  
        - ylabel (str): _description_. y轴标签  
        - xlim (tuple[float, float]|None): _description_. x轴范围  
        - ylim (tuple[float, float]|None): _description_. y轴范围  
        - xscale (str | ScaleBase): _description_. x轴刻度  
        - yscale (str | ScaleBase): _description_. y轴刻度  
        - legend (list[str]|None): _description_. 图例  
    """
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(xlabel: str, ylabel: str, X, Y=None, legend: list[str]|None=None, 
         xlim: tuple[float, float]|None=None,ylim: tuple[float, float]|None=None, 
         xscale: str='linear', yscale: str='linear',fmts: tuple[str, str, str, str] =('-', 'm--', 'g-.', 'r:'),
         figsize: tuple[float, float]=(3.5, 2.5), axes: ax.Axes|None=None,
         png_path: str=r"./lab_img/output.png"):
    """_summary_:
        画图函数，绘制对应的图像
        
    Args:
        - xlabel (str): _description_. Defaults to None. x轴标签.  
        - ylabel (str): _description_. Defaults to None. y轴标签.
        - X (_type_): _description_. 绘制的X轴数据.  
        - Y (_type_, optional): _description_. Defaults to None. 绘制的Y轴数据.    
        - legend (list[str] | None, optional): _description_. Defaults to None. 图例.  
        - xlim (tuple[float, float] | None, optional): _description_. Defaults to None. x轴范围.   
        - ylim (tuple[float, float] | None, optional): _description_. Defaults to None. y轴范围.  
        - xscale (str, optional): _description_. Defaults to 'linear'. x轴刻度.  
        - yscale (str, optional): _description_. Defaults to 'linear'. y轴刻度.  
        - fmts (tuple[str, str, str, str], optional): _description_. Defaults to ('-', 'm--', 'g-.', 'r:'). 绘制的格式.  
        - figsize (tuple[float, float], optional): _description_. Defaults to (3.5, 2.5) 图像大小.
        - axes (ax.Axes | None, optional): _description_. Defaults to None. 坐标轴.
        - png_path (str, optional): _description_. Defaults to "./lab_img/output.png". 保存的png图片路径.
    """
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X) -> bool:
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    # 标准化X和Y的格式
    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    plt.savefig(fname=png_path)
    plt.close()
    
if __name__ == "__main__":
    x = np.arange(0, 3, 0.1)
    x.reshape(2, -1)
    plot(X = x, Y = x, xlabel='x', ylabel='y', legend=['y'])
    