#!/bin/bash
rm -rf res.csv

python jaffe_test.py -d py_cuda -b 500
python coil_20_test.py -d py_cuda -b 500
python kth_test.py -d py_cuda -b 100
python uiuc_test.py -d py_cuda -b 250
python cifar_test.py -d py_cuda -b 100
python mnist_test.py -d py_cuda -b 100

python jaffe_test.py -d numba_cuda -b 500
python coil_20_test.py -d numba_cuda -b 500
python kth_test.py -d numba_cuda -b 100
python uiuc_test.py -d numba_cuda -b 250
python cifar_test.py -d numba_cuda -b 100
python mnist_test.py -d numba_cuda -b 100
