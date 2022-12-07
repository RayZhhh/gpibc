import gzip
import time

import numpy as np
import argparse

import utils
import sys
sys.path.append('./')
from gpibc.classifier import BinaryClassifier, BinaryClassifierWithInstanceSelection

data_path = '../datasets/mnist/train-images-idx3-ubyte.gz'
label_path = '../datasets/mnist/train-labels-idx1-ubyte.gz'
test_data_path = '../datasets/mnist/t10k-images-idx3-ubyte.gz'
test_label_path = '../datasets/mnist/t10k-labels-idx1-ubyte.gz'


def _load_img():
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28, 28)  # 共image_size = 28 * 28列
    return data


def _load_label():
    with gzip.open(label_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


def _load_test_img():
    with gzip.open(test_data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28, 28)  # 共image_size = 28 * 28列
    return data


def _load_test_label():
    with gzip.open(test_label_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


datasets_path = 'datasets/mnist/img_'
test_set_path = 'datasets/mnist/test_'


def _create_mnist_file():
    dataset, label = _load_img(), _load_label()
    for i in range(len(dataset)):
        stored_path = datasets_path + str(label[i]) + '.txt'
        with open(stored_path, 'a') as out:
            img = dataset[i].reshape(1, -1).squeeze()
            for i in range(len(img)):
                out.write(str(img[i]))
                if i != len(img) - 1:
                    out.write(' ')
                else:
                    out.write('\n')


def _create_mnist_test_file():
    dataset, label = _load_test_img(), _load_test_label()
    for i in range(len(dataset)):
        stored_path = test_set_path + str(label[i]) + '.txt'
        with open(stored_path, 'a') as out:
            img = dataset[i].reshape(1, -1).squeeze()
            for i in range(len(img)):
                out.write(str(img[i]))
                if i != len(img) - 1:
                    out.write(' ')
                else:
                    out.write('\n')


def print_mnist(img):
    for i in range(28):
        for j in range(28):
            print(img[i][j], end=' ')
        print()


def load_mnist_of_label(label):
    ret = np.array([])
    stored_path = datasets_path + str(label) + '.txt'
    with open(stored_path, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            l = line.split(' ')
            l = [int(p) for p in l]
            l = np.array(l).astype(float)
            ret = np.append(ret, l)
    return ret.reshape(-1, 28, 28)


def load_test_set_of_label(label):
    ret = np.array([])
    stored_path = test_set_path + str(label) + '.txt'
    with open(stored_path, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            l = line.split(' ')
            l = [int(p) for p in l]
            l = np.array(l).astype(float)
            ret = np.append(ret, l)
    return ret.reshape(-1, 28, 28)


def create_dataset_and_label(l1, l2):
    print('loading dataset...')
    data1 = load_mnist_of_label(l1)
    data2 = load_mnist_of_label(l2)
    data_ret = np.append(data1, data2)
    data_ret = data_ret.reshape(-1, 28, 28)
    label = np.array([1 for _ in range(len(data1))] + [-1 for _ in range(len(data2))])
    return data_ret, label


def create_test_set_and_label(l1, l2):
    print('loading test set...')
    data1 = load_test_set_of_label(l1)
    data2 = load_test_set_of_label(l2)
    data_ret = np.append(data1, data2)
    data_ret = data_ret.reshape(-1, 28, 28)
    label = np.array([1 for _ in range(len(data1))] + [-1 for _ in range(len(data2))])
    return data_ret, label


def test_mnist(l1, l2, eval_batch, device):
    dataset, label = create_dataset_and_label(l1, l2)
    dataset, label = utils.shuffle_dataset_and_label(dataset, label)
    test_data, test_label = create_test_set_and_label(l1, l2)
    print(f'dataset shape: {dataset.shape}')
    print(f'test data shape: {test_data.shape}')

    with open('../res.csv', 'a') as fout:
        fout.write('mnist_test\n')
        for _ in range(10):
            classifier = BinaryClassifier(dataset, label, test_data, test_label, device=device,
                                          eval_batch=eval_batch)

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
    parser.add_argument('--batch', '-b', default=5)
    parser.add_argument('--label1', '-l1', default=0)
    parser.add_argument('--label2', '-l2', default=1)
    parser.add_argument('--device', '-d', default='py_cuda')
    device = parser.parse_args().device
    eval_batch = int(parser.parse_args().batch)
    l1 = int(parser.parse_args().label1)
    l2 = int(parser.parse_args().label2)
    print(f'eval_batch: {eval_batch}; l1: {l1}; l2: {l2}')

    test_mnist(l1, l2, eval_batch=eval_batch, device=device)
