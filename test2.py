import time

from PIL import Image
import os

from gpibc.evaluator import *
from gpibc.classifier import BinaryClassifier

IH = 128
IW = 128


def create_dataset():
    data_ret = np.array([])
    label_ret = np.array([])
    for root, ds, fs in os.walk('jaffe/happiness'):
        num = 0
        for f in fs:
            image = Image.open('jaffe/happiness/' + f).convert('L')  # 用PIL中的Image.open打开图像
            image = image.resize((IH, IW))
            image_arr = np.array(image)  # 转化成numpy数组
            image_arr = image_arr.astype(float)
            data_ret = np.append(data_ret, image_arr)
            label_ret = np.append(label_ret, [1])

            num += 1
            # if num >= 10:
            #     break

    for root, ds, fs in os.walk('jaffe/surprise'):
        num = 0
        for f in fs:
            image = Image.open('jaffe/surprise/' + f).convert('L')  # 用PIL中的Image.open打开图像
            image = image.resize((IH, IW))
            image_arr = np.array(image)  # 转化成numpy数组
            image_arr = image_arr.astype(float)
            data_ret = np.append(data_ret, image_arr)
            label_ret = np.append(label_ret, [-1])

            num += 1
            # if num >= 10:
            #     break

    return data_ret.reshape(-1, IH, IW), label_ret


if __name__ == '__main__':
    data, label = create_dataset()

    classifier = BinaryClassifier(data, label)
    ts = time.time()
    classifier.train()
    print('training time: ', time.time() - ts)