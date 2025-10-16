import time
from ..src.utils.timer import Timer

def test_timer():
    t = Timer()
    t.start()
    time.sleep(1)
    print(t.stop())
    t.start()
    time.sleep(1.5)
    print(t.stop())
    t.start()
    time.sleep(2)
    print(t.stop())
    print(t.sum())
    print(t.avg())
    print(t.cumsum())