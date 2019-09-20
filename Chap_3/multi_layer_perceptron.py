# -*- coding: utf-8 -*-
# Author  :   Ellen Song
# Contact :   bili_code@163.com
# Project :   Learning-d2lai-pytorch
# File    :   multi_layer_perceptron.py
# Time    :   2019.09.20
# Desc    :   Multi layer preception with pytorch.

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils import data as Data


def init(batch_size):
  train_set = torchvision.datasets.FashionMNIST(r"D:\Dataset",
                                                train=True,
                                                transform=transforms.Compose([transforms.ToTensor()]),
                                                download=False)
  val_set = torchvision.datasets.FashionMNIST(r"D:\Dataset",
                                              train=False,
                                              transform=transforms.Compose([transforms.ToTensor()]),
                                              download=False)
  train_loader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
  val_loader = Data.DataLoader(val_set, batch_size=1, shuffle=True)

  return train_loader, val_loader


class MLP(torch.nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    self.model = torch.nn.Sequential(
      torch.nn.Linear(28 * 28 * 1, 256),
      torch.nn.ReLU(),
      torch.nn.Linear(256, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, 10)
    )
    for params in self.model.parameters():
      torch.nn.init.normal_(params, mean=0, std=0.01)

  def forward(self, x):
    x = x.view(x.shape[0], -1)
    return self.model(x)


def main():
  batch_size = 32
  train_gen, val_gen = init(batch_size)
  model = MLP()
  loss = torch.nn.CrossEntropyLoss()
  opt = torch.optim.Adam(params=model.parameters(), lr=0.0001)
  x_axis = list()
  y_axis = list()

  for idx, (x, y) in enumerate(train_gen):
    y_pred = model(x)
    ls = loss(y_pred, y)
    print("第{}次训练，方差为{}".format(idx, ls))
    opt.zero_grad()
    ls.backward()
    opt.step()
    x_axis.append(idx)
    y_axis.append(ls)

  plt.plot(x_axis, y_axis)
  plt.show()


if __name__ == '__main__':
  main()