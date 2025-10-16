from ..src.utils import display
import numpy as np

# 测试utiles/display.py中的use_svg_display()函数
def test_use_svg_display():
    display.use_TKAgg_display()
    print("test_svg_display passed")
    

# 测试utiles/display.py中的set_figsize()函数
def test_set_figsize():
    display.set_figsize()
    print("test_set_figsize passed")
    
# 尝试画图
def test_plot():
    x = np.arange(0, 3, 0.1)
    display.plot(x, [3 * x ** 2 - 4 * x, 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
    