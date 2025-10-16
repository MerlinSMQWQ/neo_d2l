import time
import numpy as np

class Timer:
    """记录多次运行的时间"""
    def __init__(self) -> None:
        """_summary_:
            用一个list存储每次运行所花费的时间, 并在初始化的时候就开始计时
        """
        self.times: list[float] = []
        self.start()
        
    def start(self) -> None:
        """_summary_:
            开始计时
        """
        self.tik: float = time.time()
        
    def stop(self) -> float:
        """_summary_:
            结束计时, 并将本次运行所花费的时间添加到times中, 并将本次运行所花费的时间存放到times中

        Returns:
            - float: _description_. 本次运行所花费的时间
        """
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def sum(self) -> float:
        """_summary_:
            返回所有运行所花费的时间

        Returns:
            - float: _description_. 所有运行花费的时间
        """
        return sum(self.times)
    
    def avg(self) -> float:
        """_summary_:
            返回平均每次运行所花费的时间

        Returns:
            - float: _description_. 平均每次运行所花费的时间
        """
        return self.sum() / len(self.times)
    
    def cumsum(self) -> list[float]:
        """_summary_:
            返回累计运行所花费的时间

        Returns:
            - list[float]: _description_. 累计运行所花费的时间, 例如times 为 [1.0, 2.0, 3.0, 4.0], cumsum() 为 [1.0, 3.0, 6.0, 10.0], 可以反映时间的的趋势
        """
        return np.array(self.times).cumsum().tolist()