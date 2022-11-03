import time

from PIL import Image
import os

from gpibc.classifier import BinaryClassifier
import numpy as np

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
            # if num >= 10:
            #     break
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
            # if num >= 10:
            #     break
    return data_ret.reshape(-1, IH, IW), label_ret


if __name__ == '__main__':
    data, label = create_dataset()
    classifier = BinaryClassifier(data, label, eval_batch=100, population_size=100, device='cuda:0')

    ts = time.time()
    classifier.train()
    print('training time: ', time.time() - ts)
