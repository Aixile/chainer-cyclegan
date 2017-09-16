import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import function
from chainer.utils import type_check
from .ops import *

class ResNetImageTransformer(chainer.Chain):
    def __init__(self, base_channels=32,  use_bn=True, normal_init=0.02, down_layers=2,
            res_layers=9, up_layers=2, upsampling='up_subpixel'):
        layers = {}
        self.down_layers = down_layers
        self.res_layers = res_layers
        self.up_layers = up_layers
        self.base_channels = base_channels

        if isinstance(use_bn, tuple) and use_bn[0] == 'multi_node_bn':
            norm = use_bn
            w = chainer.initializers.Normal(normal_init)
        elif use_bn:
            norm = 'bn'
            w = chainer.initializers.Normal(normal_init)
        else:
            norm = None
            w = None

        base = base_channels
        layers['c_first'] =  NNBlock(3, base, nn='conv', k_size=7, norm=norm, w_init=w)
        for i in range(self.down_layers):
            layers['c_down_'+str(i)] = NNBlock(base, base*2, nn='down_conv', norm=norm, w_init=w)
            base = base * 2
        for i in range(self.res_layers):
            layers['c_res_'+str(i)] = ResBlock(base, norm=norm)
        for i in range(self.up_layers):
            layers['c_up_'+str(i)] = NNBlock(base, base//2, nn='up_subpixel', norm=norm, w_init=w)
            base = base // 2
        layers['c_last'] =  NNBlock(base, 3, nn='conv', k_size=7, norm=None, w_init=w, activation=F.tanh)

        super(ResNetImageTransformer, self).__init__(**layers)

    def __call__(self, x):
        h = self.c_first(x)
        for i in range(self.down_layers):
            h = getattr(self, 'c_down_'+str(i))(h)
        for i in range(self.res_layers):
            h = getattr(self, 'c_res_'+str(i))(h)
        for i in range(self.up_layers):
            h = getattr(self, 'c_up_'+str(i))(h)
        h = self.c_last(h)
        return h
