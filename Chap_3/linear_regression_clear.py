# -*- coding: utf-8 -*-
# Author  :   Ellen Song
# Contact :   bili_code@163.com
# Project :   Learning-d2lai-pytorch
# File    :   linear_regression_clear.py
# Time    :   2019.09.16
# Desc    :   Linear regression with pytorch.

import torch

from matplotlib import pyplot as plt
from torch.utils import data as Data


def init():
  # y = 3 * x1 - 4 * x2 + 9
  x_train = torch.randn(10000, 2)
  w_train = torch.Tensor([[3], [-4]])
  b_train = torch.Tensor([[9]])
  y_train = torch.mm(x_train, w_train) + b_train
  datasets = Data.TensorDataset(x_train, y_train)
  return datasets

class LRModel(torch.nn.Module):
  def __init__(self):
    super(LRModel, self).__init__()
    self.model = torch.nn.Linear(2, 1)

  def forward(self, x):
    y = self.model(x)
    return y


def main():
  batch_size = 32
  lr = 0.3
  x_axis = list()
  y_axis = list()

  model = LRModel().cuda()
  datasets = init()
  data_loader = Data.DataLoader(datasets,
                                batch_size=batch_size,
                                shuffle=True)
  mse_loss = torch.nn.MSELoss(reduction='mean')
  optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)

  epoch = 0
  for x_true, y_true in data_loader:
    x_true = x_true.cuda()
    y_true = y_true.cuda()
    y_pred = model(x_true)
    loss = mse_loss(y_true, y_pred)
    print("第{}次训练，方差为{}".format(epoch, loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    x_axis.append(epoch)
    y_axis.append(loss)
    epoch += 1
  plt.plot(x_axis, y_axis)
  plt.show()
  x_val = torch.Tensor([[0.6, 0.2]])
  y_val = model(x_val.cuda())
  print(y_val)


if __name__ == '__main__':
  main()