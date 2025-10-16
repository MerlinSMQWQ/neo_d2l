import matplotlib
import matplotlib.pyplot as plt

# 
def use_svg_display() -> None:
    """_summary_: 使用svg, 运行直接显示, 不依赖jupyter
    """
    matplotlib.use('SVG')
    # 使用交互模式
    plt.ion()

def set_figsize(figsize: tuple[float, float] = (3.5, 2.5)) -> None:
    """_summary_

    Args:
        figsize (tuple[float, float], optional): _description_. Defaults to (3.5, 2.5).
    """
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize