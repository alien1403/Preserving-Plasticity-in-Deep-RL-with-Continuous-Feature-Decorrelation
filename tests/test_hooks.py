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

def test_hook_manager_diagnostics():
    model = SimpleMLP()
    hook_mgr = DiagnosticHookManager(model)
    dummy_input = torch.randn(64, 16)
    _ = model(dummy_input)
    stats = hook_mgr.compute_diagnostics(clear_after=True)
    assert "fc1/effective_rank" in stats and "fc1/dormant_ratio" in stats
    assert "fc1/frobenius_norm" in stats and "fc2/stable_rank" in stats
    assert len(hook_mgr.activations) == 0
    hook_mgr.remove_hooks()
