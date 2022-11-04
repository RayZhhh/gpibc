import time

from PIL import Image
import os

from gpibc.classifier import BinaryClassifier
import numpy as np

IH = 32
IW = 32


def create_dataset(name1, name2):
    print('Create dataset.')
    data_ret = np.array([])
    label_ret = np.array([])
    # load label [1]
    for root, ds, fs in os.walk('datasets/cifar/cifar_' + name1):
        for f in fs:
            image = Image.open('datasets/cifar/cifar_' + name1 + '/' + f).convert('L')  # 用PIL中的Image.open打开图像
            image = image.resize((IH, IW))
            image_arr = np.array(image)  # 转化成numpy数组
            image_arr = image_arr / 255
            image_arr = image_arr.astype(float)
            data_ret = np.append(data_ret, image_arr)
            label_ret = np.append(label_ret, [1])
    # load label [-1]
    for root, ds, fs in os.walk('datasets/cifar/cifar_' + name2):
        for f in fs:
            image = Image.open('datasets/cifar/cifar_' + name2 + '/' + f).convert('L')  # 用PIL中的Image.open打开图像
            image = image.resize((IH, IW))
            image_arr = np.array(image)  # 转化成numpy数组
            image_arr = image_arr.astype(float)
            image_arr = image_arr / 255
            data_ret = np.append(data_ret, image_arr)
            label_ret = np.append(label_ret, [-1])
    print('Create dataset OK.')
    return data_ret.reshape(-1, IH, IW), label_ret


def create_test_dataset(name1, name2):
    print('Create test set.')
    data_ret = np.array([])
    label_ret = np.array([])
    # load label [1]
    for root, ds, fs in os.walk('datasets/cifar/test_' + name1):
        for f in fs:
            image = Image.open('datasets/cifar/test_' + name1 + '/' + f).convert('L')  # 用PIL中的Image.open打开图像
            image = image.resize((IH, IW))
            image_arr = np.array(image)  # 转化成numpy数组
            image_arr = image_arr / 255
            image_arr = image_arr.astype(float)
            data_ret = np.append(data_ret, image_arr)
            label_ret = np.append(label_ret, [1])
    # load label [-1]
    for root, ds, fs in os.walk('datasets/cifar/test_' + name2):
        for f in fs:
            image = Image.open('datasets/cifar/test_' + name2 + '/' + f).convert('L')  # 用PIL中的Image.open打开图像
            image = image.resize((IH, IW))
            image_arr = np.array(image)  # 转化成numpy数组
            image_arr = image_arr.astype(float)
            image_arr = image_arr / 255
            data_ret = np.append(data_ret, image_arr)
            label_ret = np.append(label_ret, [-1])
    print('Create dataset OK.')
    return data_ret.reshape(-1, IH, IW), label_ret


if __name__ == '__main__':
    data, label = create_dataset('dog', 'cat')
    test_data, test_label = create_test_dataset('dog', 'cat')
    print(data.shape)

    classifier = BinaryClassifier(data, label, test_data, test_label, eval_batch=15, device='cuda:0')

    ts = time.time()
    classifier.train()
    print('training time: ', time.time() - ts)
    classifier.run_test()
