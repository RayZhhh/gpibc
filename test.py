import time

import numpy as np
from PIL import Image
import os
from evaluator import *

from numba import cuda

from classifier import BinaryClassifier
from program import Program
from tree import Node


def create_dataset():
    data_ret = np.array([])
    label_ret = np.array([])
    for root, ds, fs in os.walk('jaffe/happiness'):
        num = 0
        for f in fs:
            image = Image.open('jaffe/happiness/' + f).convert('L')  # 用PIL中的Image.open打开图像
            image = image.resize((128, 128))
            image_arr = np.array(image)  # 转化成numpy数组
            image_arr = image_arr.astype(float)
            data_ret = np.append(data_ret, image_arr)
            label_ret = np.append(label_ret, [1])

            num += 1
            if num >= 10:
                break

    for root, ds, fs in os.walk('jaffe/surprise'):
        num = 0
        for f in fs:
            image = Image.open('jaffe/surprise/' + f).convert('L')  # 用PIL中的Image.open打开图像
            image = image.resize((128, 128))
            image_arr = np.array(image)  # 转化成numpy数组
            image_arr = image_arr.astype(float)
            data_ret = np.append(data_ret, image_arr)
            label_ret = np.append(label_ret, [-1])

            num += 1
            if num >= 10:
                break

    return data_ret.reshape(-1, 128, 128), label_ret


data, label = create_dataset()
print(data)
print(label)

for i in range(500):
    program = Program(128, 128, init_method='growth')
    # print(program)
    fit = GPUProgramEvaluator(data, label)
    fit.fitness_evaluate(program)
    # print(program.fitness)


ts = time.time()
for i in range(500):
    program = Program(128, 128, init_method='growth')
    # print(program)
    fit = GPUProgramEvaluator(data, label)
    fit.fitness_evaluate(program)
    # print(program.fitness)
print('time: ', time.time() - ts)