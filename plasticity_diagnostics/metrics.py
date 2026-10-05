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

def compute_stable_rank(activations: torch.Tensor, eps: float = 1e-10) -> float:
    if activations.ndim > 2:
        activations = activations.reshape(activations.shape[0], -1)
    centered = activations - activations.mean(dim=0, keepdim=True)
    fro_norm_sq = torch.sum(centered ** 2)
    try:
        s = torch.linalg.svdvals(centered.to(torch.float32))
        s_max_sq = (s[0] ** 2) if len(s) > 0 else torch.tensor(eps)
        if s_max_sq < eps: return 1.0
        return float((fro_norm_sq / s_max_sq).item())
    except Exception:
        return 1.0
from typing import Tuple

def compute_dormant_neurons(activations: torch.Tensor, tau: float = 0.05, eps: float = 1e-8) -> Tuple[float, torch.Tensor]:
    if activations.ndim > 2:
        activations = activations.mean(dim=(2, 3))
    mean_act = torch.mean(torch.abs(activations), dim=0)
    layer_mean = torch.mean(mean_act)
    if layer_mean < eps:
        dormant_mask = torch.ones(mean_act.shape[0], dtype=torch.bool, device=activations.device)
        return 1.0, dormant_mask
    normalized_scores = mean_act / (layer_mean + eps)
    dormant_mask = normalized_scores <= tau
    return float(dormant_mask.float().mean().item()), dormant_mask

def compute_dead_relus(activations: torch.Tensor, eps: float = 1e-8) -> float:
    if activations.ndim > 2:
        activations = activations.reshape(activations.shape[0], -1)
    max_acts = torch.max(torch.abs(activations), dim=0)[0]
    return float((max_acts < eps).float().mean().item())
