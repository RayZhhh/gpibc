import gzip
import numpy as np

from gpibc.classifier import BinaryClassifier

data_path = 'datasets/mnist/train-images-idx3-ubyte.gz'
label_path = 'datasets/mnist/train-labels-idx1-ubyte.gz'


def _load_img():
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28, 28)  # 共image_size = 28 * 28列
    return data


def _load_label():
    with gzip.open(label_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


datasets_path = 'datasets/mnist/img_'


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


def printmnist(img):
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


def create_dataset_and_label(l1, l2):
    print('loading dataset...')
    data1 = load_mnist_of_label(l1)
    data2 = load_mnist_of_label(l2)
    data_ret = np.append(data1, data2) / 255
    data_ret = data_ret.reshape(-1, 28, 28)
    label = np.array([1 for _ in range(len(data1))] + [-1 for _ in range(len(data2))])
    return data_ret, label


def test_mnist():
    dataset, label = create_dataset_and_label(1, 2)
    print(f'dataset shape: {dataset.shape} \n label shape: {label.shape}')
    classifier = BinaryClassifier(dataset, label, eval_batch=15)
    classifier.train()


if __name__ == '__main__':
    test_mnist()

