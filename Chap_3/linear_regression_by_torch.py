# -*- coding: utf-8 -*-
# Copyright 2018 Huairuo.ai.
# Author: Song Jiaqi(song.jiaqi@huairuo.ai)

import torch

from matplotlib import pyplot as plt
from torch.autograd import Variable


def init():
  # y = 3 * x1 - 4 * x2 + 9
  x_train = torch.randn(10000, 2)
  w_train = torch.Tensor([[3], [-4]])
  b_train = torch.Tensor([[9]])
  y_train = torch.mm(x_train, w_train) + b_train
  return x_train, y_train


def gen(x_data, y_data, batch_size):
  assert len(x_data) == len(y_data), "Wrong length of datasets."
  datasets = torch.cat((x_data, y_data), dim=1)
  for i in range(0, len(datasets), batch_size):
    x = datasets[i: i + batch_size, :-1]
    y = datasets[i: i + batch_size, -1]
    y = torch.unsqueeze(y, dim=1)
    yield x, y


def squared_loss(y_true, y_pred):
  loss = (y_pred - y_true).pow(2) / 2
  return loss


def bp(w, b, lr, batch_size):
  w.data -= w.grad * lr / batch_size
  b.data -= b.grad * lr / batch_size
  return w, b


def main():
  x_train, y_train = init()
  batch_size = 32
  lr = 0.3
  w = Variable(torch.randn(2, 1), requires_grad=True)
  b = Variable(torch.randn(1, 1), requires_grad=True)
  x_axis = list()
  y_axis = list()
  epoch = 0
  for x_true, y_true in gen(x_train, y_train, batch_size):
    y_pred = torch.mm(x_true, w) + b
    loss = squared_loss(y_pred, y_true).sum() / batch_size
    loss.backward()
    w, b = bp(w, b, lr, batch_size)
    print("第{}次训练，方差为{}".format(epoch, loss))
    w.grad.data.zero_()
    b.grad.data.zero_()
    x_axis.append(epoch)
    y_axis.append(loss)
    epoch += 1
  plt.plot(x_axis, y_axis)
  plt.show()
  x_val = torch.Tensor([[0.6, 0.2]])
  y_val = torch.mm(x_val, w) + b
  print(y_val)


if __name__ == '__main__':
  main()
