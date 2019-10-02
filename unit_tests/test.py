import torch
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
import cv2
import numpy as np
from skimage import io, feature, transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_edge_map(img, scaling = None):
    h, w = img.shape[0], img.shape[1]
    if scaling:
        img = transform.resize(img, (scaling * h, scaling * w), anti_aliasing = True)
    edges = 255 * feature.canny(img, sigma = 2).astype(np.uint8)
    if scaling:
        edges = transform.resize(edges, (h, w), anti_aliasing = True, preserve_range = True).astype(np.uint8)
        edges = cv2.equalizeHist(edges)
    return edges

trsfm = transforms.Compose([transforms.ToTensor()])
img = Image.open('1.png')
x = trsfm(img)
gray = x[0] * 0.2126 + x[1] * 0.7152 + x[2] * 0.0722
save_image(gray, 'gray.png')

edge = get_edge_map(gray, 4)
plt.imshow(edge, cmap=plt.cm.gray)
plt.savefig('edge.png')