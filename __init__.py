__version__ = '0.0.1'

from .src.utils.display import use_TKAgg_display, set_figsize, set_axes, plot
from .src.utils.math import linreg, squared_loss
from .src.utils.quick_gen import synthetic_data
from .src.utils.optimizer import sgd
from .src.utils.timer import Timer

__all__ = [
    'use_TKAgg_display',
    'set_figsize',
    'set_axes',
    'plot',
    'linreg',
    'squared_loss',
    'synthetic_data',
    'sgd',
    'Timer'
]