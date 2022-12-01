#!/bin/bash
rm -rf res.csv
python jaffe_test.py -b 500
python cifar_test.py -b 100
python coil_20_test.py -b 500
python kth_test.py -b 100
python mnist_test.py -b 100
python uiuc_test.py -b 250