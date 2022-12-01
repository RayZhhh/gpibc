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
    for root, ds, fs in os.walk('datasets/jaffe/happiness'):
        num = 0
        for f in fs:
            image = Image.open('datasets/jaffe/happiness/' + f).convert('L')  # 用PIL中的Image.open打开图像
            image = image.resize((IH, IW))
            image_arr = np.array(image)  # 转化成numpy数组
            image_arr = image_arr.astype(float)
            if num >= 8:
                train_data = np.append(train_data, image_arr)
                train_label = np.append(train_label, [1])
            else:
                test_data = np.append(test_data, image_arr)
                test_label = np.append(test_label, [1])
            num += 1

    # load label [-1]
    for root, ds, fs in os.walk('datasets/jaffe/surprise'):
        num = 0
        for f in fs:
            image = Image.open('datasets/jaffe/surprise/' + f).convert('L')  # 用PIL中的Image.open打开图像
            image = image.resize((IH, IW))
            image_arr = np.array(image)  # 转化成numpy数组
            image_arr = image_arr.astype(float)
            if num >= 7:
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

    with open('res.csv', 'a') as fout:
        fout.write('jaffe_test\n')
        for _ in range(5):
            classifier = BinaryClassifier(traind, trainl, testd, testl, device='py_cuda', eval_batch=250)
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

            del(classifier)
        fout.write('\n')
