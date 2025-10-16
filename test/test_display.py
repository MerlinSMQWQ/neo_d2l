from ..src.utils import display


# 测试utiles/display.py中的use_svg_display()函数
def test_use_svg_display():
    display.use_svg_display()
    print("test_svg_display passed")
    

# 测试utiles/display.py中的set_figsize()函数
def test_set_figsize():
    display.set_figsize()
    print("test_set_figsize passed")