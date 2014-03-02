#!/bin/sh
# 16mb, already split into train/valid/test and pickled
mkdir -p /home/panic/Desktop/mnist
cd /home/panic/Desktop/mnist
wget http://deeplearning.net/data/mnist/mnist.pkl.gz
