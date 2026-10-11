import pytest
import torch
import torch.nn as nn
from plasticity_diagnostics.hooks import DiagnosticHookManager

class SimpleMLP(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32, out_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def test_hook_registration():
    model = SimpleMLP()
    hook_mgr = DiagnosticHookManager(model)
    assert len(hook_mgr.hooks) == 2
    dummy_input = torch.randn(64, 16)
    _ = model(dummy_input)
    assert "fc1" in hook_mgr.activations and "fc2" in hook_mgr.activations
    assert hook_mgr.activations["fc1"].shape == (64, 32)
    hook_mgr.remove_hooks()
