import pytest
import torch
from plasticity_diagnostics.loss import FeatureDecorrelationLoss

def test_decorrelation_loss_ortho_vs_correlated():
    torch.manual_seed(42)
    b, d = 200, 32
    loss_fn = FeatureDecorrelationLoss()
    independent_features = torch.randn(b, d)
    loss_independent = loss_fn(independent_features).item()
    assert loss_independent < 0.05
    base_col = torch.randn(b, 1)
    correlated_features = base_col.repeat(1, d) + 1e-4 * torch.randn(b, d)
    loss_correlated = loss_fn(correlated_features).item()
    assert loss_correlated > 0.80
