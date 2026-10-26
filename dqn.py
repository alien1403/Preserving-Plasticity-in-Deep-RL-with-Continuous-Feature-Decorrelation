import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor):
        h1 = self.relu1(self.fc1(x))
        h2 = self.relu2(self.fc2(h1))
        q_vals = self.head(h2)
        return q_vals, h2
