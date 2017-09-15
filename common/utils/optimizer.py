import numpy as np
import math
import chainer
try:
    import chainermn
except:
    pass

def make_adam(model, lr=0.0002, beta1=0.9, beta2=0.999):
    optimizer = chainer.optimizers.Adam(alpha=lr, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    return optimizer

def make_adam_mn(model, comm, lr=0.0002, beta1=0.9, beta2=0.999):
    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.Adam(alpha=lr, beta1=beta1, beta2=beta2), comm)
    optimizer.setup(model)
    return optimizer

def make_rmsprop(model, comm, lr=0.0002, alpha = 0.99):
    optimizer = chainer.optimizers.RMSprop(lr=lr, alpha=alpha)
    optimizer.setup(model)
    return optimizer

def make_rmsprop_mn(model, lr=0.0002, alpha = 0.99):
    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.RMSprop(lr=lr, alpha=alpha), comm)
    optimizer.setup(model)
    return optimizer
