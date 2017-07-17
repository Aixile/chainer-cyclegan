import numpy as np
import sys, os
#sys.path.append(os.path.dirname(__file__))

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import Variable

from chainer import cuda
from chainer import serializers
import numpy as np
from chainer import Variable
import chainer
import math

def add_noise(h, test, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if test:
        return h
    else:
        return h + sigma * xp.random.randn(*h.data.shape)

class ResBlock(chainer.Chain):
    def __init__(self, ch, bn=True, activation=F.relu):
        self.bn = bn
        self.activation = activation
        layers = {}
        layers['c0'] = L.Convolution2D(ch, ch, 3, 1, 1)
        layers['c1'] = L.Convolution2D(ch, ch, 3, 1, 1)
        if bn:
            layers['bn0'] = L.BatchNormalization(ch)
            layers['bn1'] = L.BatchNormalization(ch)
        super(ResBlock, self).__init__(**layers)

    def __call__(self, x, test):
        h = self.c0(x)
        if self.bn:
            h = self.bn0(h, test=test)
        h = self.activation(h)
        h = self.c1(h)
        if self.bn:
            h = self.bn1(h, test=test)
        return h + x


class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.relu, dropout=False, noise=False):
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        self.sample = sample
        self.noise = noise
        layers = {}
        w = chainer.initializers.Normal(0.02)
        if sample=='down':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        elif sample=='none-9':
            layers['c'] = L.Convolution2D(ch0, ch1, 9, 1, 4, initialW=w)
        elif sample=='none-7':
            layers['c'] = L.Convolution2D(ch0, ch1, 7, 1, 3, initialW=w)
        elif sample=='none-5':
            layers['c'] = L.Convolution2D(ch0, ch1, 5, 1, 2, initialW=w)
        else:
            layers['c'] = L.Convolution2D(ch0, ch1, 3, 1, 1, initialW=w)
        if bn:
            if self.noise:
                layers['batchnorm'] = L.BatchNormalization(ch1, use_gamma=False)
            else:
                layers['batchnorm'] = L.BatchNormalization(ch1)
        super(CBR, self).__init__(**layers)

    def __call__(self, x, test):
        if self.sample=="down" or self.sample=="none" or self.sample=='none-9' or self.sample=='none-7' or self.sample=='none-5':
            h = self.c(x)
        elif self.sample=="up":
            h = F.unpooling_2d(x, 2, 2, 0, cover_all=False)
            h = self.c(h)
        else:
            print("unknown sample method %s"%self.sample)
        if self.bn:
            h = self.batchnorm(h, test=test)
        if self.noise:
            h = add_noise(h, test=test)
        if self.dropout:
            h = F.dropout(h, train=not test)
        if not self.activation is None:
            h = self.activation(h)
        return h


class Generator_ResBlock_6(chainer.Chain):
    def __init__(self):
        super(Generator_ResBlock_6, self).__init__(
            c1 = CBR(3, 32, bn=True, sample='none-7'),
            c2 = CBR(32, 64, bn=True, sample='down'),
            c3 = CBR(64, 128, bn=True, sample='down'),
            c4 = ResBlock(128, bn=True),
            c5 = ResBlock(128, bn=True),
            c6 = ResBlock(128, bn=True),
            c7 = ResBlock(128, bn=True),
            c8 = ResBlock(128, bn=True),
            c9 = ResBlock(128, bn=True),
            c10 = CBR(128, 64, bn=True, sample='up'),
            c11 = CBR(64, 32, bn=True, sample='up'),
            c12 = CBR(32, 3, bn=True, sample='none-7', activation=F.tanh)
        )

    def __call__(self, x, test=False, volatile=False):
        h = self.c1(x, test=test)
        h = self.c2(h, test=test)
        h = self.c3(h, test=test)
        h = self.c4(h, test=test)
        h = self.c5(h, test=test)
        h = self.c6(h, test=test)
        h = self.c7(h, test=test)
        h = self.c8(h, test=test)
        h = self.c9(h, test=test)
        h = self.c10(h, test=test)
        h = self.c11(h, test=test)
        h = self.c12(h, test=test)
        return h

class Generator_ResBlock_9(chainer.Chain):
    def __init__(self):
        super(Generator_ResBlock_9, self).__init__(
            c1 = CBR(3, 32, bn=True, sample='none-7'),
            c2 = CBR(32, 64, bn=True, sample='down'),
            c3 = CBR(64, 128, bn=True, sample='down'),
            c4 = ResBlock(128, bn=True),
            c5 = ResBlock(128, bn=True),
            c6 = ResBlock(128, bn=True),
            c7 = ResBlock(128, bn=True),
            c8 = ResBlock(128, bn=True),
            c9 = ResBlock(128, bn=True),
            c10 = ResBlock(128, bn=True),
            c11 = ResBlock(128, bn=True),
            c12 = ResBlock(128, bn=True),
            c13 = CBR(128, 64, bn=True, sample='up'),
            c14 = CBR(64, 32, bn=True, sample='up'),
            c15 = CBR(32, 3, bn=True, sample='none-7', activation=F.tanh)
        )

    def __call__(self, x, test=False, volatile=False):
        h = self.c1(x, test=test)
        h = self.c2(h, test=test)
        h = self.c3(h, test=test)
        h = self.c4(h, test=test)
        h = self.c5(h, test=test)
        h = self.c6(h, test=test)
        h = self.c7(h, test=test)
        h = self.c8(h, test=test)
        h = self.c9(h, test=test)
        h = self.c10(h, test=test)
        h = self.c11(h, test=test)
        h = self.c12(h, test=test)
        h = self.c13(h, test=test)
        h = self.c14(h, test=test)
        h = self.c15(h, test=test)
        return h


class Discriminator(chainer.Chain):
    def __init__(self, in_ch=3, n_down_layers=4):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        self.n_down_layers = n_down_layers

        layers['c0'] = CBR(in_ch, 64, bn=False, sample='down', activation=F.leaky_relu, dropout=False, noise=True)
        base = 64

        for i in range(1, n_down_layers):
            layers['c'+str(i)] = CBR(base, base*2, bn=True, sample='down', activation=F.leaky_relu, dropout=False, noise=True)
            base*=2

        layers['c'+str(n_down_layers)] = CBR(base, 1, bn=False, sample='none', activation=None, dropout=False, noise=True)

        super(Discriminator, self).__init__(**layers)

    def __call__(self, x_0, test=False):
        h = self.c0(x_0, test=test)

        for i in range(1, self.n_down_layers+1):
            h = getattr(self, 'c'+str(i))(h, test=test)

        return h
