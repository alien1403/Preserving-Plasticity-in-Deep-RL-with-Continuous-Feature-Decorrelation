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
