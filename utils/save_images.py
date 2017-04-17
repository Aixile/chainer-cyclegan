import cv2
#from PIL import Image
import numpy as np
from chainer import cuda
import chainer
import cupy
import os


def copy_to_cpu(imgs):
    if type(imgs) == chainer.variable.Variable :
        imgs = imgs.data
    if type(imgs) == cupy.core.core.ndarray:
        imgs = cuda.to_cpu(imgs)
    return imgs

def post_processing_tanh(imgs):
    imgs = (imgs + 1) *127.5
    imgs = np.clip(imgs, 0, 255)
    imgs = imgs.astype(np.uint8)
    return imgs

# Input imgs format: (batch, channels, width, height)
def save_images_grid(imgs, path, grid_w=4, grid_h=4, post_processing=post_processing_tanh):
    imgs = copy_to_cpu(imgs)
    if post_processing is not None:
        imgs = post_processing(imgs)
    b, ch, w, h = imgs.shape
    assert b == grid_w*grid_h

    imgs = imgs.reshape((grid_w, grid_h, ch, w, h))
    imgs = imgs.transpose(0, 1, 3, 4, 2)
    imgs = imgs.reshape((grid_w, grid_h, w, h, ch)).transpose(0, 2, 1, 3, 4).reshape((grid_w*w, grid_h*h, ch))
    if ch==1:
        imgs = imgs.reshape((grid_w*w, grid_h*h))
    cv2.imwrite(path, imgs)
    #Image.fromarray(imgs[:,:,::-1]).save(path)
