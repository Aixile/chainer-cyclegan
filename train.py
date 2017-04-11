#!/usr/bin/env python

# python train.py --batchsize 8 --gpu 3 --out result

from __future__ import print_function
import argparse
import os

import chainer
from chainer import training
from chainer.training import extension
from chainer.training import extensions

import sys
#sys.path.append(os.path.dirname(__file__)+os.path.sep+os.path.pardir)

import net
from updater import *
from evaluation import *

from chainer import cuda, serializers
import pickle
from horse2zebra import *

#from common.lsun_bedroom_dataset import LSUN_Bedroom_Dataset
#import common.paths as paths

def main():
    parser = argparse.ArgumentParser(
        description='Train script for test NN module')
    parser.add_argument('--batchsize', '-b', type=int, default=16)
    parser.add_argument('--max_iter', '-m', type=int, default=120000)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--eval_folder', '-e', default='test',
                        help='Directory to output the evaluation result')

    parser.add_argument('--eval_interval', type=int, default=100,
                        help='Interval of evaluating generator')

    parser.add_argument("--learning_rate_g", type=float, default=0.0002, help="Learning rate for generator")
    parser.add_argument("--learning_rate_d", type=float, default=0.0002, help="Learning rate for discriminator")

    parser.add_argument("--load_gen_model", default='', help='load generator model')
    parser.add_argument("--load_dis_model", default='', help='load discriminator model')

    parser.add_argument('--gen_class', default='Generator_ResBlock_9', help='Default generator class')
    parser.add_argument('--dis_class', default='Discriminator', help='Default discriminator class')

    parser.add_argument("--lambda1", type=float, default=1.0, help='lambda for reconstruction loss')
    parser.add_argument("--lambda2", type=float, default=10.0, help='lambda for adversarial loss')
#    parser.add_argument("--lambda3", type=float, default=0.0, help='lambda for feature loss')
#    parser.add_argument("--lambda4", type=float, default=0.0, help='lambda for total variation loss')

    args = parser.parse_args()
    print(args)

    max_iter = args.max_iter

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()

    gen_g = getattr(net, args.gen_class)()
    dis_x = getattr(net, args.dis_class)()
    gen_f = getattr(net, args.gen_class)()
    dis_y = getattr(net, args.dis_class)()

    #scr = Scribbler(n_input=1)
    #vgg = pickle.load(open(paths.pretrained_vgg_imagenet, 'rb'))
    #dis  = Discriminator(in_ch=3)


    #if args.load_gen_model != '':
    #    serializers.load_npz(args.load_gen_model, scr)
    #    print("Generator model loaded")

    #if args.load_dis_model != '':
    #    serializers.load_npz(args.load_dis_model, dis)
    #    print("Discriminator model loaded")


    if not os.path.exists(args.eval_folder):
         os.makedirs(args.eval_folder)

    # select GPU
    if args.gpu >= 0:
        gen_g.to_gpu()
        gen_f.to_gpu()
        dis_x.to_gpu()
        dis_y.to_gpu()
        print("use gpu {}".format(args.gpu))

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        return optimizer

    opt_g=make_optimizer(gen_g, alpha=args.learning_rate_g)
    opt_f=make_optimizer(gen_f, alpha=args.learning_rate_g)
    opt_x=make_optimizer(dis_x, alpha=args.learning_rate_d)
    opt_y=make_optimizer(dis_y, alpha=args.learning_rate_d)

    train_dataset = horse2zebra_Dataset_train()
    train_iter = chainer.iterators.MultiprocessIterator(
        train_dataset, 1, n_processes=4)

    train2_iter = chainer.iterators.MultiprocessIterator(
        train_dataset, args.batchsize, n_processes=4)

    test_dataset = horse2zebra_Dataset_test()
    test_iter = chainer.iterators.SerialIterator(
        test_dataset, 4)

    # Set up a trainer
    updater = Updater(
        models=(gen_g, gen_f, dis_x, dis_y),
        iterator={
            'main': train_iter,
            'dis' : train2_iter,
            'test': test_iter
            },
        optimizer={
            'gen_g': opt_g,
            'gen_f': opt_f,
            'dis_x': opt_x,
            'dis_y': opt_y
            },
        device=args.gpu,
        params={
            'lambda1': args.lambda1,
            'lambda2': args.lambda2,
        #    'lambda3': args.lambda3,
            'image_size' : 256
            #'lambda4': args.lambda4,
        })

    model_save_interval = (4000, 'iteration')
    trainer = training.Trainer(updater, (max_iter, 'iteration'), out=args.out)
    trainer.extend(extensions.snapshot_object(
        gen_g, 'gen_g{.updater.iteration}.npz'), trigger=model_save_interval)
    trainer.extend(extensions.snapshot_object(
        gen_f, 'gen_f{.updater.iteration}.npz'), trigger=model_save_interval)
    trainer.extend(extensions.snapshot_object(
        dis_x, 'dis_x{.updater.iteration}.npz'), trigger=model_save_interval)
    trainer.extend(extensions.snapshot_object(
        dis_y, 'dis_y{.updater.iteration}.npz'), trigger=model_save_interval)

    log_keys = ['epoch', 'iteration', 'gen_g/loss_rec', 'gen_f/loss_rec', 'gen_g/loss_adv', 'gen_f/loss_adv', 'dis_x/loss', 'dis_y/loss']
    trainer.extend(extensions.LogReport(keys=log_keys, trigger=(20, 'iteration')))
    trainer.extend(extensions.PrintReport(log_keys), trigger=(20, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=50))

    trainer.extend(
        evaluation(gen_g, gen_f, args.eval_folder
        ), trigger=(args.eval_interval ,'iteration')
    )

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
