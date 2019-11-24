import torch.nn as nn
import torch.nn.functional as F

class NodeModule(nn.Module):
  def __init__(self):
    super(NodeModule, self).__init__()
    self.fc0 = nn.Linear(in_features=512, out_features=512)
    self.fc1 = nn.Linear(in_features=512, out_features=254)

  def forward(self, x):
    x = F.relu(self.fc0(x))
    return self.fc1(x)


class EdgeModule(nn.Module):
  def __init__(self):
    super(EdgeModule, self).__init__()
    self.fc0 = nn.Linear(in_features=512, out_features=512)
    self.fc1 = nn.Linear(in_features=512, out_features=256)
  def forward(self, x):
    x = F.relu(self.fc0(x))
    return self.fc1(x)