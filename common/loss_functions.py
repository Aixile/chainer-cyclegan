import numpy as np
import chainer
import chainer.functions as F

def loss_l1(h, t):
    return F.sum(F.absolute(h-t)) / np.prod(h.data.shape)

def loss_l2(h, t):
    return F.sum((h-t)**2) / np.prod(h.data.shape)

def loss_l2_norm(h, t, axis=(1)):
    return F.sum(F.sqrt(F.sum((h-t)**2, axis=axis))) / h.data.shape[0]

def loss_func_dcgan_dis_real(y_real):
    return F.sum(F.softplus(-y_real)) / np.prod(y_real.data.shape)

def loss_func_dcgan_dis_fake(y_fake):
    return F.sum(F.softplus(y_fake)) / np.prod(y_fake.data.shape)

def loss_func_lsgan_dis_real(y_real, label=0.9):
    return loss_l2(y_real, label)

def loss_func_lsgan_dis_fake(y_fake, label=0.1):
    return loss_l2(y_fake, label)

def loss_func_tv_l1(x_out):
    xp = cuda.get_array_module(x_out.data)
    b, ch, h, w = x_out.data.shape
    Wx = xp.zeros((ch, ch, 2, 2), dtype="f")
    Wy = xp.zeros((ch, ch, 2, 2), dtype="f")
    for i in range(ch):
        Wx[i,i,0,0] = -1
        Wx[i,i,0,1] = 1
        Wy[i,i,0,0] = -1
        Wy[i,i,1,0] = 1
    return F.sum(F.absolute(F.convolution_2d(x_out, W=Wx))) + F.sum(F.absolute(F.convolution_2d(x_out, W=Wy)))

def loss_func_tv_l2(x_out):
    xp = cuda.get_array_module(x_out.data)
    b, ch, h, w = x_out.data.shape
    Wx = xp.zeros((ch, ch, 2, 2), dtype="f")
    Wy = xp.zeros((ch, ch, 2, 2), dtype="f")
    for i in range(ch):
        Wx[i,i,0,0] = -1
        Wx[i,i,0,1] = 1
        Wy[i,i,0,0] = -1
        Wy[i,i,1,0] = 1
    return F.sum(F.convolution_2d(x_out, W=Wx) ** 2) + F.sum(F.convolution_2d(x_out, W=Wy) ** 2)

def loss_sigmoid_cross_entropy_with_logits(x, t):
    return F.average(x - x*t + F.softplus(-x)) # / x.data.shape[0])
