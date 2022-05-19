import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Policy(nn.Module):
  def __init__(self, in_space, out_space, lr=0.0005):
    super(Policy, self).__init__()
    self.fc1 = nn.Linear(in_space.shape[0], 24)
    self.fc2 = nn.Linear(24, 24)
    self.fc3 = nn.Linear(24, out_space.n)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x)) 
    x = self.fc3(x)
    return x

  def backprop(self, loss):
    self.optimizer.zero_grad()
    loss.backward(retain_graph=True)
    self.optimizer.step()
    return loss

class DynamicsModel(nn.Module):
  def __init__(self, in_space, out_space, lr=0.0005):
    super(DynamicsModel, self).__init__()
    self.fc1 = nn.Linear(in_space.shape[0]+out_space.n, in_space.shape[0]+out_space.n )
    self.fc2 = nn.Linear(in_space.shape[0]+out_space.n, in_space.shape[0]+out_space.n )
    self.fc3 = nn.Linear(in_space.shape[0]+out_space.n, in_space.shape[0]+out_space.n )
    self.fc4 = nn.Linear(in_space.shape[0]+out_space.n, in_space.shape[0])
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, s, a):
    x = torch.cat([s,a], dim=0)
    x = F.relu(self.fc1(x)) 
    x = F.relu(self.fc2(x)) 
    x = F.relu(self.fc3(x)) 
    x = self.fc4(x)
    return x

  def backprop(self, loss):
    self.optimizer.zero_grad()
    loss.backward(retain_graph=True)
    self.optimizer.step()
    return loss

class RewardsModel(nn.Module):
  def __init__(self, in_space, out_space, lr=0.0005):
    super(RewardsModel, self).__init__()
    self.fc1 = nn.Linear(in_space.shape[0]+out_space.n, in_space.shape[0]+out_space.n )
    self.fc2 = nn.Linear(in_space.shape[0]+out_space.n, in_space.shape[0]+out_space.n )
    self.fc4 = nn.Linear(in_space.shape[0]+out_space.n, 1)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, s, a):
    x = torch.cat([s,a], dim=0)
    x = F.relu(self.fc1(x)) 
    x = F.relu(self.fc2(x)) 
    x = self.fc4(x)
    return x

  def backprop(self, loss):
    self.optimizer.zero_grad()
    loss.backward(retain_graph=True)
    self.optimizer.step()
    return loss
