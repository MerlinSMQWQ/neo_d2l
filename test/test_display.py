from ..src.utils import display
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import multinomial
import torch

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
    display.plot(X = x, Y =[3 * x ** 2 - 4 * x, 2 * x - 3], xlabel='x', ylabel='f(x)', legend=['f(x)', 'Tangent line (x=1)'])
    
def test_plot_2():
    fair_probs = torch.ones([6]) / 6
    multinomial.Multinomial(1, fair_probs).sample()
    counts = multinomial.Multinomial(10, fair_probs).sample((500,))
    cum_counts = counts.cumsum(dim=0)
    estimates = cum_counts / cum_counts.sum(dim=1, keepdim=True)

    display.set_figsize((6, 4.5))
    for i in range(6):
        plt.plot(estimates[:, i].numpy(),
                label=("P(die=" + str(i + 1) + ")"))
    plt.axhline(y=0.167, color='black', linestyle='dashed')
    plt.gca().set_xlabel('Groups of experiments')
    plt.gca().set_ylabel('Estimated probability')
    plt.legend()
    plt.show()
    plt.close()
    