import os
import sys

root_horse2zebra = '/home/aixile/Workspace/dataset/horse2zebra/'
horse2zebra_trainA_key = 'keys_trainA.txt'
horse2zebra_trainB_key = 'keys_trainB.txt'
horse2zebra_testA_key = 'keys_testA.txt'
horse2zebra_testB_key = 'keys_testB.txt'

def readAllString(file):
    with open(file,'r') as f:
        lines = f.readlines()
    return [i.strip() for i in lines]
