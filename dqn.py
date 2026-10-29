import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from plasticity_diagnostics.metrics import compute_effective_rank

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

class DQNAgent:
    def __init__(self, obs_dim: int, action_dim: int, lr: float = 1e-3, gamma: float = 0.99, buffer_size: int = 20_000, batch_size: int = 64, device: str = "cpu"):
        self.device = torch.device(device)
        self.q_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.buffer = deque(maxlen=buffer_size)

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon: return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals, _ = self.q_net(obs_t)
            return int(torch.argmax(q_vals, dim=-1).item())

    def store_transition(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
