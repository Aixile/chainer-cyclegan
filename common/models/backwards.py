import numpy as np
import math
import chainer
import chainer.functions as F
from chainer import cuda, optimizers, serializers, Variable

def backward_linear(x_in, x, l):
    y = F.matmul(x, l.W)
    return y

def backward_convolution(x_in, x, l):
    y = F.deconvolution_2d(x, l.W, None, l.stride, l.pad, None)#(x_in.data.shape[2], x_in.data.shape[3]))
    return y

def backward_deconvolution(x_in, x, l):
    y = F.convolution_2d(x, l.W, None, l.stride, l.pad)
    return y

def backward_layernormalization(x_in, x, l):
    vx = Variable(x_in.data)
    y = l(vx)
    y.grad = l.xp.ones_like(y.data)
    y.backward()
    return vx.grad * x

def backward_relu(x_in, x):
    y = (x_in.data>0) * x
    return y

def backward_leaky_relu(x_in, x, a=0.2):
    y = (x_in.data>0) * x + a * (x_in.data<0) * x
    return y

def backward_tanh(x_in, x):
    y = (1.0 - x_in**2) * x
    return y

def backward_sigmoid(x_in, x):
    y = (x_in - x_in**2) * x
    return y
