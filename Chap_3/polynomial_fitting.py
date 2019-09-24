# AUTHOR: BiliCoder
# CONTACT: bili_code@163.com
# DATETIME: 2019/9/24 20:01

import matplotlib.pylab as plt
import torch
import torch.utils.data as Data


def init(batch_size):
  # y = 1.2 * x ^ 3 - 3.4 * x ^ 2 + 5.6 * x * 3 + 5 + epsilon
  x = torch.rand((100000, 1))
  x_train = torch.cat((torch.pow(x, 3), torch.pow(x, 2), torch.pow(x, 3)),
                      dim=1)
  w = torch.Tensor([[1.2], [3.4], [5.6]])
  b = torch.Tensor([5])
  y_train = torch.add(torch.mm(x_train, w), b)
  dataset = Data.TensorDataset(x, y_train)
  data_loader = Data.DataLoader(dataset, batch_size, shuffle=True)
  return data_loader


def gen(x, y, batch_size):
  dataset = Data.TensorDataset(x, y)
  data_loader = Data.DataLoader(dataset, batch_size, shuffle=True)
  return data_loader


class PFModel(torch.nn.Module):
  def __init__(self):
    super(PFModel, self).__init__()
    self.model = torch.nn.Sequential(
      torch.nn.Linear(1, 1),
    )

  def forward(self, x):
    return self.model(x)


def train():
  x_axis = list()
  y_axis = list()
  batch_size = 32
  lr = 0.01
  gen = init(batch_size)
  model = PFModel()
  loss = torch.nn.MSELoss(reduction='mean')
  opt = torch.optim.Adam(model.parameters(), lr=lr)

  for idx, (x_train, y_train) in enumerate(gen):
    y_pred = model(x_train)
    opt.zero_grad()
    ls = loss(y_pred, y_train)
    print("第{}次训练的loss为: {}".format(idx, ls))
    ls.backward()
    opt.step()
    x_axis.append(idx)
    y_axis.append(ls)

  plt.plot(x_axis, y_axis)
  plt.show()


if __name__ == '__main__':
  train()
