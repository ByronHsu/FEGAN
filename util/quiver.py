import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import torchvision.transforms as transforms 

def plot_quiver(flow, n_sample = 30, _range = [-1, 1]):
    """
    visualize the flow map
    args:
        flow: (H, W, 2) tensor
        n_sample: the number sampled on an axis
        _range: the range of the map
    """
    size = flow.shape[0]
    indexs = np.arange(0, size, size / n_sample, np.int_)

    L, R = _range
    X = np.arange(L, R, (R - L) / len(indexs))
    Y = np.arange(L, R, (R - L) / len(indexs))
    flow = flow.cpu().detach().numpy()
    x_indexs, y_indexs = np.meshgrid(indexs, indexs)
    U, V = flow[x_indexs, y_indexs, 0], flow[x_indexs, y_indexs, 1]
    fig, ax = plt.subplots()
    ax.quiver(X, Y, U, V, width = 0.003)
    fig.canvas.draw()

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data
