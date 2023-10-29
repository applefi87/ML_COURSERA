import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import fontManager
project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
font_path = os.path.join(project_dir, 'resources', 'fonts',"ChineseFont.ttf")

class DataPlotter:
    def __init__(self):
        fontManager.addfont(font_path)
        mpl.rc("font", family="ChineseFont")
    
    def scatter_plot(self, x, y, x_label, y_label, title, point_size=1):
        plt.scatter(x, y, s=point_size)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()