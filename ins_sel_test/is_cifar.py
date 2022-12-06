import argparse
import sys
import time

from PIL import Image
import os

import utils
sys.path.append('../')
from gpibc.classifier import BinaryClassifier, BinaryClassifierWithInstanceSelection
import numpy as np

IH = 32
IW = 32


def create_dataset(name1, name2):
    print('Create dataset.')
    data_ret = np.array([])
    label_ret = np.array([])
    # load label [1]
    for root, ds, fs in os.walk('../datasets/cifar/cifar_' + name1):
        num = 0
        for f in fs:
            image = Image.open('../datasets/cifar/cifar_' + name1 + '/' + f).convert('L')  # 用PIL中的Image.open打开图像
            image = image.resize((IH, IW))
            image_arr = np.array(image)  # 转化成numpy数组
            image_arr = image_arr.astype(float)
            data_ret = np.append(data_ret, image_arr)
            label_ret = np.append(label_ret, [1])
            num += 1
    # load label [-1]
    for root, ds, fs in os.walk('../datasets/cifar/cifar_' + name2):
        num = 0
        for f in fs:
            image = Image.open('../datasets/cifar/cifar_' + name2 + '/' + f).convert('L')  # 用PIL中的Image.open打开图像
            image = image.resize((IH, IW))
            image_arr = np.array(image)  # 转化成numpy数组
            image_arr = image_arr.astype(float)
            data_ret = np.append(data_ret, image_arr)
            label_ret = np.append(label_ret, [-1])
            num += 1
    print('Create dataset OK.')
    return data_ret.reshape(-1, IH, IW), label_ret


def create_test_dataset(name1, name2):
    print('Create test set.')
    data_ret = np.array([])
    label_ret = np.array([])
    # load label [1]
    for root, ds, fs in os.walk('../datasets/cifar/test_' + name1):
        for f in fs:
            image = Image.open('../datasets/cifar/test_' + name1 + '/' + f).convert('L')  # 用PIL中的Image.open打开图像
            image = image.resize((IH, IW))
            image_arr = np.array(image)  # 转化成numpy数组
            image_arr = image_arr.astype(float)
            data_ret = np.append(data_ret, image_arr)
            label_ret = np.append(label_ret, [1])
    # load label [-1]
    for root, ds, fs in os.walk('../datasets/cifar/test_' + name2):
        for f in fs:
            image = Image.open('../datasets/cifar/test_' + name2 + '/' + f).convert('L')  # 用PIL中的Image.open打开图像
            image = image.resize((IH, IW))
            image_arr = np.array(image)  # 转化成numpy数组
            image_arr = image_arr.astype(float)
            data_ret = np.append(data_ret, image_arr)
            label_ret = np.append(label_ret, [-1])
    print('Create test set OK.')
    return data_ret.reshape(-1, IH, IW), label_ret


def run_cifar(l1, l2, eval_batch, device, ins_sel):
    data, label = create_dataset(l1, l2)
    data, label = utils.shuffle_dataset_and_label(data, label)
    test_data, test_label = create_test_dataset(l1, l2)
    print(f'data.shape: {data.shape}')
    print(f'test_data.shape: {test_data.shape}')

    with open('ISS_res.csv', 'a') as fout:
        fout.write('cifar_test\n')
        for _ in range(10):
            classifier = BinaryClassifierWithInstanceSelection(data, label, test_data, test_label, device=device, eval_batch=eval_batch)

            # train
            ts = time.time()
            classifier.train()
            dur = time.time() - ts
            print('training time: ', dur)
            print('fit eval time: ', classifier.fitness_evaluation_time)

            # test
            classifier.run_test()

            # write result
            fout.write(str(dur) + ',' + str(classifier.best_test_program.fitness) + ',' + str(
                classifier.fitness_evaluation_time) + '\n')

            del (classifier)
        fout.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for mnist test.')
    parser.add_argument('--batch', '-b', default=10)
    parser.add_argument('--label1', '-l1', default='dog')
    parser.add_argument('--label2', '-l2', default='cat')
    parser.add_argument('--device', '-d', default='cpu')
    parser.add_argument('--instance_select', '-i', default='False')
    eval_batch = int(parser.parse_args().batch)
    device = parser.parse_args().device
    instance_select = bool(parser.parse_args().instance_select)
    l1 = parser.parse_args().label1
    l2 = parser.parse_args().label2

    print(f'eval_batch: {eval_batch}; l1: {l1}; l2: {l2}')

    run_cifar(l1, l2, eval_batch, device, instance_select)
