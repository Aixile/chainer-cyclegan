import cv2
import numpy as np
from chainer import cuda
import chainer
try:
    import cupy
except:
    pass
import os

def copy_to_cpu(imgs):
    if type(imgs) == chainer.variable.Variable :
        imgs = imgs.data
    try:
        if type(imgs) == cupy.core.core.ndarray:
            imgs = cuda.to_cpu(imgs)
    except:
        pass
    return imgs

def postprocessing_tanh(imgs):
    imgs = (imgs + 1) * 127.5
    imgs = np.clip(imgs, 0, 255)
    imgs = imgs.astype(np.uint8)
    return imgs

def save_single_image(img, path, post_processing=postprocessing_tanh):
    img = copy_to_cpu(img)
    if post_processing is not None:
        img = post_processing(img)
    #ch, w, h = img.shape
    img = img.transpose((1, 2, 0))
    cv2.imwrite(path, img)

def save_images_grid(imgs, path, grid_w=4, grid_h=4, post_processing=postprocessing_tanh, transposed=False):
    imgs = copy_to_cpu(imgs)
    if post_processing is not None:
        imgs = post_processing(imgs)
    b, ch, w, h = imgs.shape
    assert b == grid_w*grid_h

    imgs = imgs.reshape((grid_w, grid_h, ch, w, h))
    imgs = imgs.transpose(0, 1, 3, 4, 2)
    if transposed:
        imgs = imgs.reshape((grid_w, grid_h, w, h, ch)).transpose(1, 2, 0, 3, 4).reshape((grid_h*w, grid_w*h, ch))
    else:
        imgs = imgs.reshape((grid_w, grid_h, w, h, ch)).transpose(0, 2, 1, 3, 4).reshape((grid_w*w, grid_h*h, ch))
    if ch==1:
        imgs = imgs.reshape((grid_w*w, grid_h*h))
    cv2.imwrite(path, imgs)
