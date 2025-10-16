import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axes as ax
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
        figsize (tuple[float, float], optional): _description_. Defaults to (3.5, 2.5).
    """
    use_TKAgg_display()
    plt.rcParams['figure.figsize'] = figsize
    
def set_axes(axes: ax.Axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend) -> None:
    """_summary_:
        设置图像坐标轴
    Args:
        axes (ax.Axes): _description_  
        xlabel (_type_): _description_  
        ylabel (_type_): _description_  
        xlim (_type_): _description_  
        ylim (_type_): _description_  
        xscale (_type_): _description_  
        yscale (_type_): _description_  
        legend (_type_): _description_  
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

def plot(X, Y=None, xlabel: str|None=None, ylabel: str|None=None, legend: list[str]|None=None, xlim=None,
         ylim=None, xscale: str='linear', yscale: str='linear',
         fmts: tuple[str, str, str, str] =('-', 'm--', 'g-.', 'r:'), figsize: tuple[float, float]=(3.5, 2.5), axes: ax.Axes|None=None):
    """_summary_:
        画图函数，绘制对应的图像
        
    Args:
        X (_type_): _description_  
        Y (_type_, optional): _description_. Defaults to None.  
        xlabel (str | None, optional): _description_. Defaults to None.  
        ylabel (str | None, optional): _description_. Defaults to None.  
        legend (list[str] | None, optional): _description_. Defaults to None.  
        xlim (_type_, optional): _description_. Defaults to None.  
        ylim (_type_, optional): _description_. Defaults to None.  
    """
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X) -> bool:
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

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
    plt.show()
    