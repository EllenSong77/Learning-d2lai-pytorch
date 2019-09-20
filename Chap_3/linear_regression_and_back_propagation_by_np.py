# -*- coding: utf-8 -*-
# Author  :   Ellen Song
# Contact :   bili_code@163.com
# Project :   Learning-d2lai-pytorch
# File    :   linear_regression_and_back_propagation_by_np.py
# Time    :   2019.09.16
# Desc    :   Linear regression and back propagation with numpy.


import numpy as np
import torch

from matplotlib import pyplot as plt


def init():
  # y = 3 * x1 - 4 * x2 + 9
  x_train = np.random.randn(10000, 2)
  w_train = [3, -4]
  b_train = 9
  y_train = w_train[0] * x_train[:, 0] + w_train[1] * x_train[:, 1] + b_train
  y_train = y_train[:, np.newaxis]
  return x_train, y_train


def gen(x_data, y_data, batch_size):
  assert len(x_data) == len(y_data), "Wrong length of datasets."
  datasets = np.concatenate((x_data, y_data), axis=1)
  np.random.shuffle(datasets)
  for i in range(0, len(datasets), batch_size):
    x = datasets[i: i + batch_size, :-1]
    y = datasets[i: i + batch_size, -1]
    y = y[:, np.newaxis]
    yield x, y


def back_propagation(w, b, x, y, lr, batch_size):
  w = w - np.dot((np.dot(x, w.T) + b - y).T, x) * lr / batch_size
  b = b - np.sum((np.dot(x, w.T) + b - y).T) * lr / batch_size
  return w, b


def squared_loss(y_true, y_pred):
  loss = (y_pred - y_true) ** 2 / 2
  return loss


def main():
  x_train, y_train = init()
  batch_size = 32
  lr = 0.01
  w = np.random.randn(1, 2)
  b = np.random.randn(1, 1)
  x_axis = list()
  y_axis = list()
  epoch = 0
  for x_true, y_true in gen(x_train, y_train, batch_size):
    y_pred = np.dot(x_true, w.T) + b
    loss = np.average(squared_loss(y_true, y_pred), axis=(0, 1))
    print("第{}次训练，方差为{}".format(epoch, loss))
    w, b = back_propagation(w, b, x_true, y_true, lr, batch_size)
    x_axis.append(epoch)
    y_axis.append(loss)
    epoch += 1
  plt.plot(x_axis, y_axis)
  plt.show()
  x_val = np.array([0.6, 0.2])
  y_val = np.dot(x_val, w.T) + b
  print(y_val)


if __name__ == '__main__':
  main()
