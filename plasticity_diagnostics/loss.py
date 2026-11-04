import torch
import torch.nn as nn

class FeatureDecorrelationLoss(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim > 2: z = z.reshape(z.shape[0], -1)
        b, d = z.shape
        if b <= 1 or d <= 1: return torch.tensor(0.0, device=z.device, requires_grad=True)
        z_cent = z - z.mean(dim=0, keepdim=True)
        cov = (z_cent.T @ z_cent) / (b - 1)
        std = torch.sqrt(torch.diag(cov).clamp(min=self.eps))
        corr = cov / (std.unsqueeze(0) * std.unsqueeze(1) + self.eps)
        eye = torch.eye(d, device=z.device, dtype=torch.bool)
        off_diag = corr[~eye]
        return torch.mean(off_diag ** 2)
