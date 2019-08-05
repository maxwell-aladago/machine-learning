import matplotlib.pyplot as plt
import numpy as np

# Do not modify this function, it has already been coded for you
def draw_line(w, b,  min_x, max_x, style):

    x = np.array([min_x, max_x])
            
    y = (-w[0]*x-b) / w[1]
    plt.plot(x, y, style, lineWidth=2)