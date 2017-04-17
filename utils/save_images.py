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

def preprocess_tanh(imgs):
    imgs = imgs.astype(np.float32)
    imgs = imgs / 127.5 - 1
    imgs = imgs.transpose((2, 0, 1))
    return imgs

def postprocessing_tanh(imgs):
    imgs = (imgs + 1) * 127.5
    imgs = np.clip(imgs, 0, 255)
    imgs = imgs.astype(np.uint8)
    return imgs

# resize the image so that the shorter edge is resize_base the width and the height could divide divide_base
# when resize_base is 0, use the shorter edge to determine the resize_base
def resize_to_nearest_aspect_ratio(img, divide_base=4, resize_base=256):
    w, h = img.shape[0], img.shape[1]
    if w < h:
        if resize_base == 0:
            resize_base = w - w % divide_base
        s0 = resize_base
        s1 = int(h * (resize_base / w))
        s1 = s1 - s1 % divide_base
    else:
        if resize_base == 0:
            resize_base = h - h % divide_base
        s1 = resize_base
        s0 = int(w * (resize_base / h))
        s0 = s0 - s0 % divide_base
    return cv2.resize(img, (s1, s0), interpolation=cv2.INTER_AREA)


# Input imgs format: (batch, channels, width, height)
def save_images_grid(imgs, path, grid_w=4, grid_h=4, post_processing=postprocessing_tanh):
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
