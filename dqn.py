import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from plasticity_diagnostics.loss import FeatureDecorrelationLoss
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
    def __init__(self, obs_dim: int, action_dim: int, lr: float = 1e-3, gamma: float = 0.99, buffer_size: int = 20_000, batch_size: int = 64, use_covreg: bool = False, covreg_coeff: float = 0.05, device: str = "cpu"):
        self.device = torch.device(device)
        self.q_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.buffer = deque(maxlen=buffer_size)
        self.use_covreg = use_covreg
        self.covreg_coeff = covreg_coeff
        self.decorr_loss_fn = FeatureDecorrelationLoss()

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon: return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals, _ = self.q_net(obs_t)
            return int(torch.argmax(q_vals, dim=-1).item())

    def store_transition(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def update(self):
        if len(self.buffer) < self.batch_size: return 0.0, 0.0, 0.0
        batch = random.sample(self.buffer, self.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        obs_t = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs_t = torch.tensor(np.array(next_obs), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_vals, features = self.q_net(obs_t)
        state_action_values = q_vals.gather(1, actions_t)
        with torch.no_grad():
            next_q_vals, _ = self.target_net(next_obs_t)
            max_next_q = next_q_vals.max(1, keepdim=True)[0]
            target_values = rewards_t + (1 - dones_t) * self.gamma * max_next_q
        td_loss = nn.functional.mse_loss(state_action_values, target_values)

        if self.use_covreg:
            decorr_loss = self.decorr_loss_fn(features)
            total_loss = td_loss + self.covreg_coeff * decorr_loss
        else:
            decorr_loss = torch.tensor(0.0)
            total_loss = td_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        with torch.no_grad(): erank = compute_effective_rank(features)
        return total_loss.item(), decorr_loss.item(), erank

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
