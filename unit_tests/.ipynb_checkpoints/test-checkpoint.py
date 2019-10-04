import torch
import os
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
from skimage import io, feature, transform

def to_grayscale(img):
    return img[0, :, :] * 0.2126 + img[1, :, :] * 0.7152 + img[2, :, :] * 0.0722

def get_edge_map(tensor, scaling = 2):
    '''
        tensor: A batch of image tensor with shape n, c, h, w.
        scaling: Scaling factor to apply to image before computing edges.
    '''
    #h, w = img.shape[0], img.shape[1]
    n, c, h, w = tensor.shape
    edge_maps = []
    for i in range(n):
        img = to_grayscale(tensor[i, :, :, :]).numpy()
        if scaling:
            img = transform.resize(img, (scaling * h, scaling * w), anti_aliasing = True)
        edges = 255 * feature.canny(img, sigma = 2).astype(np.uint8)
        if scaling:
            edges = transform.resize(edges, (h, w), anti_aliasing = True, preserve_range = True).astype(np.uint8)
            edges = cv2.equalizeHist(edges)
            # edges = (edges != 0) * 255
        edge_maps.append(torch.tensor(edges, dtype = torch.float).unsqueeze(0).unsqueeze(0))
    return torch.cat(tuple(edge_maps), dim = 0)

def cal_smooth(flow):
    '''
        Smooth constraint for flow map
    '''
    gx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])  # NCHW
    gy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])  # NCHW
    smooth = torch.mean(gx) + torch.mean(gy)
    return smooth

def radial_constraint(flow):
    '''
        Flow map has dimension (n, h, w, 2).
    '''
    def normalize_flow(flow):
        norm = torch.sqrt(flow[:, :, :, 0] ** 2 + flow[:, :, :, 1] ** 2)
        flow /= norm.unsqueeze(3).repeat(1, 1, 1, 2)
        return flow
    
    n, h, w, _ = flow.shape
    x = torch.arange(0, h, dtype = torch.float).unsqueeze(1).repeat(1, w) - (h - 1) / 2
    y = torch.arange(0, w, dtype = torch.float).unsqueeze(0).repeat(h, 1) - (w - 1) / 2
    v = torch.cat((x.unsqueeze(2), y.unsqueeze(2)), dim = 2).unsqueeze(0).repeat(n, 1, 1, 1)
    v = normalize_flow(v)
    flow = normalize_flow(flow)
    inner_product = torch.mul(v[:, :, :, 0], flow[:, :, :, 0]) + torch.mul(v[:, :, :, 1], flow[:, :, :, 1])
    radial_loss = torch.sum(inner_product.view(n, -1), dim = 1)
    return torch.mean(radial_loss)

trsfm = transforms.Compose([transforms.ToTensor()])
img = Image.open('test.jpg')
x = trsfm(img).unsqueeze(0).repeat(5, 1, 1, 1)
edge = get_edge_map(x, 2)
save_image(edge, 'edge.jpg')
