#!/usr/bin/env python

import os
#import glob
import numpy as np
import chainer
import chainer.cuda
from chainer import cuda, serializers, Variable
from chainer import training
import chainer.functions as F
import cv2
import argparse
import common.net as net
import datasets
#from PIL import Image
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CycleGAN single image testing script')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gen_class', '-c', default='Generator_ResBlock_9', help='Default generator class')
    parser.add_argument("--load_gen_model", '-l', default='', help='load generator model')
    parser.add_argument('--input_channels',  type=int, default=3, help='number of input channels')
    parser.add_argument('--output', '-o', default='output' ,help='output image path')
    parser.add_argument('--input', '-i', default='' ,help='input image path')
    parser.add_argument("--recurrent", '-r', type=int, default=1, help='apply the function recursively')
    parser.add_argument("--base_size", '-s', type=int, default=256, help='shorter edge length')


    args = parser.parse_args()
    print(args)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()

    gen = getattr(net, args.gen_class)()

    if args.load_gen_model != '':
        serializers.load_npz(args.load_gen_model, gen)
        print("Generator model loaded")

    if args.gpu >= 0:
        gen.to_gpu()
        print("use gpu {}".format(args.gpu))

    #test_dataset = getattr(datasets, args.load_dataset)(flip=0, resize_to=args.resize_to, crop_to=args.crop_to)

    xp = gen.xp
    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    img = resize_to_nearest_aspect_ratio(img, resize_base=args.base_size)
    img = preprocess_tanh(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    if args.gpu >= 0:
        img = cuda.to_gpu(img)

    input = img
    for i in range(args.recurrent):
        output = gen(input,  volatile=True)
        output.unchain_backward()
        input = output.data


    save_images_grid(output,path=args.output+".jpg", grid_w=1, grid_h=1)
