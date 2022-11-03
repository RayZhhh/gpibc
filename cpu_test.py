import sys

import numpy as np
from PIL import Image
import os

from gpibc.eval_gpu import GPUPopulationEvaluator
from gpibc.eval_cpu import CPUEvaluator
from gpibc.program import Program

IH = 128
IW = 128


def create_dataset():
    data_ret = np.array([])
    label_ret = np.array([])
    # load label [1]
    for root, ds, fs in os.walk('datasets/jaffe/happiness'):
        num = 0
        for f in fs:
            image = Image.open('datasets/jaffe/happiness/' + f).convert('L')  # 用PIL中的Image.open打开图像
            image = image.resize((IH, IW))
            image_arr = np.array(image)  # 转化成numpy数组
            image_arr = image_arr / 255
            image_arr = image_arr.astype(float)
            data_ret = np.append(data_ret, image_arr)
            label_ret = np.append(label_ret, [1])
            num += 1
            if num >= 5:
                break
    # load label [-1]
    for root, ds, fs in os.walk('datasets/jaffe/surprise'):
        num = 0
        for f in fs:
            image = Image.open('datasets/jaffe/surprise/' + f).convert('L')  # 用PIL中的Image.open打开图像
            image = image.resize((IH, IW))
            image_arr = np.array(image)  # 转化成numpy数组
            image_arr = image_arr.astype(float)
            image_arr = image_arr / 255
            data_ret = np.append(data_ret, image_arr)
            label_ret = np.append(label_ret, [-1])
            num += 1
            if num >= 5:
                break
    return data_ret.reshape(-1, IH, IW), label_ret


if __name__ == '__main__':
    data, label = create_dataset()
    eval = CPUEvaluator(data, label)
    geval = GPUPopulationEvaluator(data, label)

    for i in range(100):
        p = Program(IH, IW, init_method='growth')
        eval.evaluate_program(p)
        cfit = p.fitness
        print('cpu: ', cfit)
        geval.evaluate_population([p])
        gfit = p.fitness
        print('gpu: ', gfit)
        print()
        if cfit != gfit:
            print(p)
            sys.exit(0)
