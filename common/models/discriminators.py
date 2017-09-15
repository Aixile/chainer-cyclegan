import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import function
from chainer.utils import type_check
from .ops import *
from .backwards import *

class DCGANDiscriminator(chainer.Chain):
    def __init__(self, in_ch=3, base_size=128, down_layers=4, use_bn=True, normal_init=0.02, noise_all_layers=False, conv_as_last=False, w_init=None):
        layers = {}

        self.down_layers = down_layers
        self.conv_as_last = conv_as_last

        if isinstance(use_bn, tuple) and use_bn[0] == 'multi_node_bn':
            norm = use_bn
        elif use_bn:
            norm = 'bn'
        else:
            norm = None

        act = F.leaky_relu
        if w_init is None:
            w_init = chainer.initializers.Normal(normal_init)

        layers['c_first'] = NNBlock(in_ch, base_size, nn='down_conv', norm=None, activation=act, noise=noise_all_layers, w_init=w_init)
        base = base_size

        for i in range(down_layers-1):
            layers['c'+str(i)] = NNBlock(base, base*2, nn='down_conv', norm=norm, activation=act, noise=noise_all_layers, w_init=w_init)
            base*=2

        if conv_as_last:
            layers['c_last'] = NNBlock(base, 1, nn='conv', norm=None, activation=None, w_init=w_init)
        else:
            layers['c_last'] = NNBlock(None, 1, nn='linear', norm=None, activation=None, w_init=w_init)

        super(DCGANDiscriminator, self).__init__(**layers)

    def __call__(self, x, retain_forward=False):
        h = self.c_first(x, retain_forward=retain_forward)
        for i in range(self.down_layers-1):
            h = getattr(self, 'c'+str(i))(h, retain_forward=retain_forward)
        if not self.conv_as_last:
            _b, _ch, _w, _h = h.data.shape
            self.last_shape=(_b, _ch, _w, _h)
            h = F.reshape(h, (_b, _ch*_w*_h))
        h = self.c_last(h, retain_forward=retain_forward)
        return h

    def clip(self, upper=0.01, lower=-0.01):
        weight_clipping(self, upper=upper, lower=lower)

    def differentiable_backward(self, g):
        g = self.c_last.differentiable_backward(g)
        if not self.conv_as_last:
            _b, _ch, _w, _h = self.last_shape
            g = F.reshape(g, (_b, _ch, _w, _h))
        for i in reversed(range(self.down_layers-1)):
            g = getattr(self, 'c'+str(i)).differentiable_backward(g)
        g = self.c_first.differentiable_backward(g)
        return g
