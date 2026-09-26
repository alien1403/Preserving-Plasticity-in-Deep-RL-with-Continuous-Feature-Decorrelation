import torch

def compute_effective_rank(activations: torch.Tensor, eps: float = 1e-10) -> float:
    if activations.ndim > 2:
        activations = activations.reshape(activations.shape[0], -1)
    B, D = activations.shape
    if min(B, D) <= 1:
        return 1.0
    centered = activations - activations.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(centered.to(torch.float32))
        s_sum = torch.sum(s)
        if s_sum < eps: return 1.0
        p = s / (s_sum + eps)
        p = p[p > eps]
        entropy = -torch.sum(p * torch.log(p))
        return float(torch.exp(entropy).item())
    except Exception:
        return 1.0
