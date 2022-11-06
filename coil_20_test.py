import time

from PIL import Image
import os

from gpibc.classifier import BinaryClassifier
import numpy as np

IH = 128
IW = 128


def create_dataset():
    train_data = np.array([])
    train_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])

    # load label [1]
    for root, ds, fs in os.walk('datasets/coil-20/'):
        num = 0
        for f in fs:
            if f.split('__')[0] == 'obj10':
                image = Image.open('datasets/coil-20/' + f).convert('L')  # 用PIL中的Image.open打开图像
                image = image.resize((IH, IW))
                image_arr = np.array(image)  # 转化成numpy数组
                image_arr = image_arr / 255
                image_arr = image_arr.astype(float)

                if num >= 18:
                    train_data = np.append(train_data, image_arr)
                    train_label = np.append(train_label, [1])
                else:
                    test_data = np.append(test_data, image_arr)
                    test_label = np.append(test_label, [1])
                num += 1

    # load label [-1]
    for root, ds, fs in os.walk('datasets/coil-20/'):
        num = 0
        for f in fs:
            if f.split('__')[0] == 'obj20':
                image = Image.open('datasets/coil-20/' + f).convert('L')  # 用PIL中的Image.open打开图像
                image = image.resize((IH, IW))
                image_arr = np.array(image)  # 转化成numpy数组
                image_arr = image_arr.astype(float)
                image_arr = image_arr / 255
                if num >= 18:
                    train_data = np.append(train_data, image_arr)
                    train_label = np.append(train_label, [-1])
                else:
                    test_data = np.append(test_data, image_arr)
                    test_label = np.append(test_label, [-1])
                num += 1
    return train_data.reshape(-1, IH, IW), train_label, test_data.reshape(-1, IH, IW), test_label


if __name__ == '__main__':
    traind, trainl, testd, testl = create_dataset()
    print(f'train data shape: {traind.shape}')
    print(f'test data shape: {testd.shape}')

    classifier = BinaryClassifier(traind, trainl, testd, testl, eval_batch=125, population_size=500, device='cuda:0')

    ts = time.time()
    classifier.train()
    print('training time: ', time.time() - ts)

    classifier.run_test()
