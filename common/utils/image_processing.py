import cv2
import numpy as np
import chainer
from chainer import cuda
try:
    import cupy
except:
    pass

def copy_to_cpu(imgs):
    if type(imgs) == chainer.variable.Variable :
        imgs = imgs.data
    try:
        if type(imgs) == cupy.core.core.ndarray:
            imgs = cuda.to_cpu(imgs)
    except:
        pass
    return imgs

# ch, w, h
def resize_dataset_image(img, out_w=256, out_h=256):
    img2 = copy_to_cpu(img)
    img2 = img2.transpose((1, 2, 0))
    img2 = cv2.resize(img2, (out_h, out_w), cv2.INTER_LANCZOS4)
    img2 = img2.transpose((2, 0, 1))

    try:
        if type(img) == cupy.core.core.ndarray:
            img2 = cuda.to_gpu(img2)
    except:
        pass

    return img2

# b, ch, w, h
def resize_batch_dataset_image(img, out_w=256, out_h=256):
    img2 = copy_to_cpu(img)
    b, ch, w, h = imgs.shape
    img2 = np.zeros((b, ch, out_w, out_h)).astype('f')
    for i in range(b):
        t = resize_dataset_image(img2[i,:], out_w, out_h)
        img2[i] = t

    try:
        if type(img) == cupy.core.core.ndarray:
            img2 = cuda.to_gpu(img2)
    except:
        pass

    return img2
