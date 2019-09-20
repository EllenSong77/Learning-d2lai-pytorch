# -*- coding: utf-8 -*-
# Author  :   Ellen Song
# Contact :   bili_code@163.com
# Project :   Learning-d2lai-pytorch
# File    :   softmax_fashion_mnist.py
# Time    :   2019.09.18
# Desc    :   Fashion MNIST softmax classification with pytorch.


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils import data as Data


def init():
  train_set = torchvision.datasets.FashionMNIST(r"D:\Dataset",
                                                train=True,
                                                transform=transforms.Compose([transforms.ToTensor()]),
                                                download=False)
  val_set = torchvision.datasets.FashionMNIST(r"D:\Dataset",
                                              train=False,
                                              transform=transforms.Compose([transforms.ToTensor()]),
                                              download=False)
  return train_set, val_set


def gen(dataset, batch_size):
  return Data.DataLoader(dataset,
                         batch_size=batch_size,
                         shuffle=True)


def softmax(x):
  x_exp = x.exp()
  exp_sum = x_exp.sum(dim=1, keepdim=True)
  return x_exp / exp_sum


class SoftmaxModel(torch.nn.Module):
  def __init__(self, batch_size):
    super(SoftmaxModel, self).__init__()
    self.batch_size = batch_size
    self.model = torch.nn.Sequential(torch.nn.Linear(28 * 28 * 1, 10))


  def forward(self, x):
    x = x.view(-1, 28 * 28 * 1)
    x = self.model(x)
    x = softmax(x)
    return x


def main():
  batch_size = 32
  y_axis = list()
  x_axis = list()
  train_set, val_set = init()
  data_loader = gen(train_set, batch_size)
  model = SoftmaxModel(batch_size)
  loss = torch.nn.CrossEntropyLoss()
  opt = torch.optim.Adam(params=model.parameters(), lr=0.0001)
  for idx, (x, y) in enumerate(data_loader):
    y_pred = model(x)
    ls = loss(y_pred, y)
    print("第{}次训练，方差为{}".format(idx, ls))
    opt.zero_grad()
    ls.backward()
    opt.step()
    y_axis.append(ls)
    x_axis.append(idx)
  plt.plot(x_axis, y_axis)
  plt.show()

  for i in range(10):
    x_val, y_val = val_set[i][0], val_set.classes[val_set[i][1]]
    y_val_pred = model(x_val)
    y_val_pred = y_val_pred.argmax(dim=1, keepdim=True)

    plt.imshow(x_val.view(28, 28).numpy())
    plt.show()
    print(y_val, val_set.classes[y_val_pred])


if __name__ == '__main__':
  main()