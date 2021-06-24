
from IPython.core.display import HTML
import cv2

def show_gif(f_path, img_width=100):
    return HTML(f'<img src="{f_path}" alt="Acoustic Camera GIF" style="width:{img_width}%"/>')