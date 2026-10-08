import pytest
import torch
from plasticity_diagnostics.metrics import compute_effective_rank

def test_effective_rank_full_vs_collapsed():
    torch.manual_seed(42)
    B, D = 100, 32
    full_rank_acts = torch.randn(B, D)
    erank_full = compute_effective_rank(full_rank_acts)
    assert erank_full > 20.0

    base_col = torch.randn(B, 1)
    rank1_acts = base_col.repeat(1, D)
    erank_rank1 = compute_effective_rank(rank1_acts)
    assert erank_rank1 == pytest.approx(1.0, abs=1e-2)

from plasticity_diagnostics.metrics import compute_stable_rank

def test_stable_rank():
    torch.manual_seed(42)
    B, D = 64, 16
    acts = torch.randn(B, D)
    srank = compute_stable_rank(acts)
    assert 1.0 <= srank <= min(B, D)

from plasticity_diagnostics.metrics import compute_dormant_neurons

def test_dormant_neurons():
    B, D = 50, 20
    acts = torch.ones(B, D)
    acts[:, :5] = 1e-4
    dormant_ratio, mask = compute_dormant_neurons(acts, tau=0.05)
    assert dormant_ratio == pytest.approx(5 / D, abs=1e-3)
    assert mask[:5].all()
    assert not mask[5:].any()

from plasticity_diagnostics.metrics import compute_dead_relus

def test_dead_relus():
    B, D = 40, 10
    acts = torch.relu(torch.randn(B, D))
    acts[:, :3] = 0.0
    dead_ratio = compute_dead_relus(acts)
    assert dead_ratio >= 0.3
import torch.nn as nn
from plasticity_diagnostics.metrics import compute_weight_metrics

def test_weight_metrics():
    linear = nn.Linear(32, 64)
    metrics = compute_weight_metrics(linear)
    assert "frobenius_norm" in metrics and "spectral_norm" in metrics
    assert metrics["frobenius_norm"] > 0.0 and metrics["spectral_norm"] > 0.0
